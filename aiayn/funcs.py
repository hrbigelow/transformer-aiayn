import functools
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
    p_logits = jax.nn.log_softmax(p_logits, axis)
    log2e = jnp.log2(jnp.exp(1.0))
    log_q = jnp.log(q)
    kl_nats = jnp.sum(safe_xy(q, log_q - p_logits), axis)
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

def gather_seqs(seqs, inds):
    """
    Inputs:
    seqs: bsq
    inds: bk

    Return:  bkq
    output[b,k,:] = seqs[b,inds[b,k],:]
    """
    B,_,Q = seqs.shape
    inds = jnp.stack(jnp.broadcast_arrays(jnp.arange(B)[:,None], inds), 2)
    gd = jax.lax.GatherDimensionNumbers((2,), (0,1), (0,1))
    return jax.lax.gather(seqs, inds, gd, (1,1,Q))

def gather_cache(cache, inds):
    """
    Gathers the same number of elements, but they may be duplicated. 
    Inputs:
    cache: lbehsqd
    inds:  be

    Return:  lbehsqd
    """
    # print(f'{cache.shape=}, {inds.shape=}')
    L,B,*_ = cache.shape
    inds = jnp.stack(jnp.broadcast_arrays(jnp.arange(B)[:,None], inds), 2)
    gd = jax.lax.GatherDimensionNumbers((0,3,4,5,6), (1,2), (1,2))
    out = jax.lax.gather(cache, inds, gd, (L,1,1,*cache.shape[3:]))
    print(f'{out.shape=}')
    return out



"""
# Pseudo-code for beam search

k = beam_size
My formulation:



E = k * 2

L: set of live sequences (not ending in EOS)
C: set of complete sequences (ending in EOS)
V: set of all tokens

Initialization:
L <- { [BOS] }
C <- { }



Assumes local and global scores are distinct
for step in 1..max_steps:
    # replace set of live sequences with topk scoring live sequences extended by 1
    L = { l+v for l in L for v in V } # concatenate a token onto each l
    L = { l for l in L if local_score(l) among top E scores }


    # add completed sequences into C
    C = C \\union { l for l in L if l[-1] == EOS }

    # replace C with top-k global scoring sequences in C
    C = { c for c in C if global_score(c) among top k scores }


    # remove completed sequences from L
    L = { l for l in L if l[-1] != EOS }
"""

def beam_search_step(eos_id, alpha, beta, beam_size, step, logits, dec_kvcache,
        xattn, live_seqs, live_scores, fin_seqs, fin_scores):
    """
    Consumes the next prediction logits
    logits:      bev   (logits for generating tokens at position step+1)
    Consumes and returns these:
    dec_kvcache: lbehsqd  (populated in [:step]
    xattn:  bet   sum of cross-attention coefficients
    live_seqs:   beq,  the set L (live sequences) 
    live_scores: be,  scores for live_seqs (-inf for non-existent)
    fin_seqs:    boq,  the set F (finished sequences)
    fin_scores:  bo,   scores (complete) for fin_seqs (-inf for non-existent)
    """
    jnp.set_printoptions(precision=2, threshold=100000, edgeitems=100, linewidth=180)
    score_fn = functools.partial(beam_search_score, alpha, beta)

    V = logits.shape[2]
    B,E,Q = live_seqs.shape
    new_scores = live_scores[:,:,None] + jax.nn.log_softmax(logits, axis=2)
    # k_prod_inds values index into the e,v space
    live_scores, k_prod_inds = jax.lax.top_k(new_scores.reshape(B,E*V), E) # bk
    # jax.debug.print('step: {}, new_scores[0,0]:\n{}', step, new_scores[0,0])
    # jax.debug.print('step: {}, live_scores[0]:\n{}', step, live_scores[0])
    k_seq_inds, k_tok_inds = jnp.divmod(k_prod_inds, V)
    # jax.debug.print('step: {}, k_seq_inds: {}', step, k_seq_inds)
    # jax.debug.print('step: {}, k_tok_inds: {}', step, k_tok_inds)
    # Could skip these gather steps if k_seq_inds is the same set as arange(E)
    live_seqs = gather_seqs(live_seqs, k_seq_inds)
    xattn = gather_seqs(xattn, k_seq_inds)
    dec_kvcache = gather_cache(dec_kvcache, k_seq_inds)
    live_seqs = live_seqs.at[:,:,step].set(k_tok_inds)
    jax.debug.print('step: {}, live_seqs:\n{}', step, live_seqs)

    any_finished = jnp.any(jnp.equal(k_tok_inds, eos_id))

    def update_fn(args):
        dec_kvcache, xattn, live_seqs, live_scores, fin_seqs, fin_scores = args
        tmp_scores = score_fn(step, live_scores, xattn)
        tmp_scores = jnp.where(live_seqs[:,:,step] == eos_id, tmp_scores, -jnp.inf)
        live_scores = jnp.where(live_seqs[:,:,step] != eos_id, live_scores, -jnp.inf)
        fin_scores = jnp.concatenate((fin_scores, tmp_scores), axis=1)
        fin_scores, fin_inds = jax.lax.top_k(fin_scores, beam_size)
        fin_seqs = gather_seqs(fin_seqs, fin_inds)
        return dec_kvcache, xattn, live_seqs, live_scores, fin_seqs, fin_scores

    def passthru_fn(args):
        return args

    args = dec_kvcache, xattn, live_seqs, live_scores, fin_seqs, fin_scores
    return jax.lax.cond(any_finished, update_fn, passthru_fn, args)

def beam_search_score(alpha, beta, out_len, scores, xattn):
    """
    Inputs:
    alpha:    float parameter
    beta:     float parameter
    out_len:  integer parameter
    scores:   bo  log(P(output|input))
    xattn:  bot  (this is sum_j { p_{ij} } in eq 14, where i := t

    Scoring function used by beam search, see eq 14 from
    https://arxiv.org/pdf/1609.08144.pdf, as referenced by original AIAYN paper.

    Author's note:
    When alpha = 0 and beta = 0, decoder falls back to pure beam search by probability
    """
    numer = (5.0 + out_len) ** alpha
    denom = 6.0 ** alpha
    lp = numer / denom
    cp = beta * jnp.log(jnp.minimum(xattn, 1.0)).sum(axis=2)
    return scores / lp + cp

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

"""
Indices Legend:
    b: batch
    o: buffer (2 * beam_size)
    e: beam (beam_size)
    v: vocab (tokens)
    q: query position (in the token sequence)
"""

def merge_extended(i, eos_id, model, ext_seq, ext_logprob, complete_seq,
        complete_scores, k):
    """
    ext_seq: boq
    ext_logprob: bo
    complete_seq: beq
    complete_scores: be

    Identifies the subset of ext_seq that end in EOS
    Computes length-normalized scores for those sequences
    Takes top k among the newly completed sequences and existing complete_seq
    Returns:

    merged_seq [batch_size, beam_size, max_seq_len]
    merged_scores [batch_size, beam_size]
    """
    current_seqlen = i + 1
    # compute adjusted scores for all ext_seq
    dec_enc_att = model.dec_enc_attention()
    ext_score = adjusted_score(alpha, beta, current_seqlen, ext_logprob, dec_enc_att)
    ext_score = jnp.where(ext_seq[:,:,i] == eos_id, -tf.inf, ext_score)
    # need to pad complete_seq

    # batch_size, 3*beam_size
    all_seq = jnp.concatenate((ext_seq, complete_seq), axis=1)
    all_score = jnp.concatenate((ext_score, complete_score), axis=1)
    new_seq, new_score = topk_seqs(all_seq, all_score, k)
    return new_seq, new_score

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

