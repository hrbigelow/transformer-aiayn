import tensorflow as tf
import jax.numpy as jnp
import jax
from jax import lax
import numpy as np

import pdb

def safe_xy(x, y):
    """
    Return 0 if x == 0, else x * y
    """
    x_ok = x != 0.
    safe_x = jnp.where(x_ok, x, 1.)
    safe_y = jnp.where(x_ok, y, 1.)
    return jnp.where(x_ok, lax.mul(safe_x, safe_y), jnp.zeros_like(x))

def entropy(p, axis):
    """
    Compute entropy in bits along axis
    """
    log2e = jnp.log2(jnp.exp(1.0))
    h_nat = jnp.sum(safe_xy(p, - jnp.log(p)), axis)
    return h_nat * log2e

def fused_kldiv_softmax(q, p_logits, axis):
    # compute D[q(x) || softmax(p_logits)] implicitly fusing the operations
    # returns value in bits
    log2e = jnp.log2(jnp.exp(1.0))
    z = jnp.max(p_logits, axis)
    scaled_p_logits = p_logits - jnp.expand_dims(z, axis)
    log_normalizer = z + jnp.log(jnp.sum(jnp.exp(scaled_p_logits), axis))
    log_normalizer = jnp.expand_dims(log_normalizer, axis)
    log_q = jnp.log(q)
    # q_entropy = - jnp.sum(jax.scipy.special.xlogy(q, q), axis)
    # cross_entropy = - (jnp.sum(q * p_logits, axis) - log_normalizer)
    kl_nats = jnp.sum(safe_xy(q, log_q - p_logits + log_normalizer), axis)
    kl = kl_nats * log2e
    # jax.debug.print("{}", kl) 
    return kl

def gather_nd(value, index, axes):
    """
    value: any shape
    index: [*dest, d]
    axes: tuple of d axes into value

    For some setting of the tuple `dest`, index[*dest,:] gives a tuple of d
    coordinates.  Combining those d coordinates with ':' in the shape of value: 

    addr = index[*dest,:]
    output[*dest] = value[interp(*addr,:)]
    """
    # check `axes` are valid relative to `value` shape
    # check `index` shape is valid relative to `axes`
    if any(a not in range(value.ndim) for a in axes):
        raise RuntimeError(
            f'axes should all be valid axes into `value`.  Got {axes=}, '
            f'{value.shape=}')
    if index.ndim == 0 or index.shape[-1] != len(axes):
        raise RuntimeError(
            f'index.shape[0] must equal number of axes.  Got {index.shape=}, '
            f'{axes=}')
    if index.dtype.is_floating_point or index.dtype.is_complex:
        raise RuntimeError(
            f'index.dtype must be integral.  Got {index.dtype=}')

    # do not check for out-of-bounds?
    num_axes = len(axes)
    perm = axes + tuple(i for i in range(value.ndim) if i not in axes)
    perm_value = value.permute(*perm)
    num_slice = np.prod(perm_value.shape[:num_axes])
    slice_shape = perm_value.shape[num_axes:]
    flat_value = perm_value.reshape(num_slice, *slice_shape)
    cumul_axes = [np.prod(axes[i+1:], initial=1) for i in range(num_axes)]
    cumul_axes = tf.constant(cumul_axes, dtype=tf.int64)
    flat_index = tf.einsum('...k,k->...', index, cumul_axes)
    out_shape = list(flat_index.shape)
    flat_result = tf.index_select(flat_value, 0, flat_index.flatten())
    result = flat_result.reshape(*out_shape, *slice_shape)
    return result

def beam_search_score(alpha, beta, out_len, log_prob, dec_enc_attn):
    """
    Scoring function used by beam search, see eq 14 from
    https://arxiv.org/pdf/1609.08144.pdf, as referenced by original AIAYN paper.
    
    dec_enc_attn: shape I,J.  p[i,j] is attention probability of j'th output token on 
                 i'th input token 
                 sum_i(p[i,j]) = 1 by construction
    log_prob: log(P(output|input)), the log probability assigned by the model

    Author's note:
    "When alpha = 0 and beta = 0, decoder falls back to pure beam search by probability"
    """
    def lp(out_len):
        # the function providing length normalization (it has no other name in the
        # paper) 
        numer = (5.0 + out_len) ** alpha
        denom = 6.0 ** alpha
        return numer / denom

    def coverage_penalty(dec_enc_attention):
        """
        dec_enc_attention: [batch, in_positions, out_positions]
        each slice [b, :, j] is normalized

        Returns:
        penalty: [batch]
        """
        term = tf.log(tf.min(dec_enc_attention.sum(axis=2), 1.0))
        return beta * term.sum(axis=1)

    return log_prob / lp(out_len) + coverage_penalty(dec_enc_attn)

def extend_paths(i, model, enc_seq, live_seq, live_logprob):
    """
    Computes the conditional log probability of each possible final token conditioned
    on all previously generated decoder sequence and the encoder sequence.

    i: new sequence position index to create prediction for
    model: conditional model log P(token | prev_seq, encoder_seq)
    live_seq: [batch_size, beam_size, max_seq_len]
       token sequences predicted by the decoder so far
       live_seq[:,:,:i] are populated, while the rest are filled with a PAD token
    live_logprob: [batch_size, beam_size]
    Returns:
    ext_logprob: [batch_size, beam_size, num_tokens]
    """
    log_prob = model(enc_seq, live_seq)
    return live_logprob * log_prob[:,i,:].unsqueeze(1)


def topk_seqs(seqs, scores, k):
    """
    seqs: [batch_size, other, context_size]
    scores: [batch_size, other]

    Return the top-k scoring seqs and scores for each batch
    """
    top_scores, inds = tf.topk(scores, k, dim=1)
    top_seqs = gather_nd(seqs, inds, axes=(1,))
    return top_seqs, top_scores

def merge_extended(i, model, ext_seq, ext_logprob, complete_seq, complete_scores, k):
    """
    ext_seq: [batch_size, 2*beam_size, max_seq_len]
    ext_logprob: [batch_size, 2*beam_size]
    complete_seq [batch_size, beam_size, max_seq_len]
    complete_scores [batch_size, beam_size] 

    Identifies the subset of ext_seq that end in EOS
    Computes length-normalized scores for those sequences
    Takes top k among the newly completed sequences and existing complete_seq
    Returns:

    merged_seq [batch_size, beam_size, max_seq_len]
    merged_scores [batch_size, beam_size]
    """
    eos_token = 5000 # FIXME
    current_seqlen = i + 1
    # compute adjusted scores for all ext_seq
    dec_enc_att = model.dec_enc_attention()
    ext_score = adjusted_score(alpha, beta, current_seqlen, ext_logprob, dec_enc_att)
    ext_score = tf.where(ext_seq[:,:,i] == eos_token, -tf.inf, ext_score)
    # need to pad complete_seq

    # batch_size, 3*beam_size
    all_seq = tf.concat((ext_seq, complete_seq), dim=1)
    all_score = tf.concat((ext_score, complete_score), dim=1)
    new_seq, new_score = topk_seqs(all_seq, all_score, k)
    return new_seq, new_score
   

"""
From the authors of https://arxiv.org/pdf/1609.08144.pdf: 

 Firstly, at each step, we only consider tokens that have local scores that are not
 more than beamsize below the best token for this step. Secondly, after a normalized
 best score has been found according to equation 14, we prune all hypotheses that are
 more than beamsize below the best normalized score so far. The latter type of
 pruning only applies to full hypotheses because it compares scores in the normalized
 space, which is only available when a hypothesis ends. This latter form of pruning
 also has the effect that very quickly no more hypotheses will be generated once a
 sufficiently good hypothesis has been found, so the search will end quickly
"""
def beam_search(model, alpha, beta, beam_size, max_length, enc_input):
    """
    """
    # initialize
    batch_size = enc_input.shape[0]
    live_seq = t.full(batch_size, 2*beam_size, max_length, pad_token_id)
    live_logprob = t.full(live_seq.shape[:2], 0.0)

    complete_seq = t.full((batch_size, beam_size, max_length), pad_token_id)
    complete_score = t.full(complete_seq.shape[:2], 0.0)

    for i in range(max_length):
        ext_logprob = extend_paths(i, model, enc_input, live_seq, live_logprob)
        ext_seq, ext_logprob = topk_seqs(live_seq, ext_logprob, 2*beam_size)
        complete_seq, complete_score = \
            merge_extended(i, model, ext_seq, ext_logprob, complete_seq,
                    complete_score, beam_size)
    return complete_seq, complete_score

