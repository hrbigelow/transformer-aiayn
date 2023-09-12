import functools
import tensorflow as tf
import jax.numpy as jnp
import jax
from jax import lax
import numpy as np

import pdb

def entropy(p, axis, where=None):
    """
    Compute entropy in bits along axis
    p: any shape including at least `axis`
    where: same shape as p
    """
    log_p = jnp.log2(jnp.where(p == 0.0, 1e-30, p))
    return jnp.sum(p * -log_p, axis=axis, where=where, initial=0.0) 

def cross_entropy(q, p_logits, axis, where=None):
    log2e = jnp.log2(jnp.exp(1.0))
    # p_logits = jax.nn.log_softmax(p_logits, axis, where=where, initial=1.0) * log2e
    p_logits = log_softmax(p_logits, axis, where, 0.0) * log2e
    return jnp.sum(q * -p_logits, axis, where=where, initial=0.0)

def fused_kldiv_softmax(q, p_logits, axis, where=None):
    # compute D[q(x) || softmax(p_logits)] implicitly fusing the operations
    # returns value in bits
    log2e = jnp.log2(jnp.exp(1.0))
    p_logits = log_softmax(p_logits, axis, where, 0.0) * log2e
    log_q = jnp.log2(jnp.where(q == 0.0, 1e-30, q))
    return jnp.sum(q * (log_q - p_logits), axis, where=where, initial=0)

def softmax(x, axis, where=None, initial=None):
    x_max = jnp.max(x, axis, where=where, initial=initial, keepdims=True)
    scaled = x - x_max
    if where is not None:
        scaled = jnp.where(where, scaled, -1000.0)
    unnormalized = jnp.exp(scaled)
    result = unnormalized / jnp.sum(unnormalized, axis, where=where, keepdims=True)
    if where is not None:
        result = jnp.where(where, result, 0.0)
    return result

def log_softmax(x, axis, where=None, initial=None):
    x_max = jnp.max(x, axis, where=where, initial=initial, keepdims=True)
    shifted = x - lax.stop_gradient(x_max)
    if where is not None:
        shifted = jnp.where(where, shifted, -1000.0)
    shifted_logsumexp = jnp.log(
            jnp.sum(jnp.exp(shifted), axis, where=where, keepdims=True))
    result = shifted - shifted_logsumexp
    if where is not None:
        return jnp.where(where, result, -jnp.inf)
    return result

def attention_entropy(attn_matrix, attn_mask, query_seqids, query_lengths,
        target_seqids, target_lengths, output_seqids, output_lengths):
    """
    computes a 0-1 scaled discrete entropy of target attention paid to a given
    sentence.  'target attention' is the attention_matrix summed over heads and
    queries consistent with the attn_mask, then normalized.

    The entropy of that marginal is taken, then scaled by log(k), where k is the
    total length of the target region as identified in target_seqids)

    Then, this scaled entropy is weighted by the length of the corresponding region
    in output_seqids (the sentence output by the decoder corresponding to this
    sentence).

    This function is called for any type of attention - Encoder self-attention,
    and Decoder cross-attention being the main use-cases.  

    For now, it seems that Decoder causally masked self-attention isn't a good
    candidate for this function.

    Finally, the average over multiple target regions is returned

    Inputs:
    attn_matrix: bhqt (softmax'ed attention - each b,h,q slice is normalized)
    attn_mask: bqt    (1 means masked, 0 means pass-through)
    query_seqids: bq  (identifies packed sentences used in attention query)
    query_lengths: bp  (p is the id dimension, indexed by target_seqids)
    target_seqids: bt (identifies packed sentences used in attention target) 
    output_seqids: bo (identifies packed output sentences corresponding to this batch)
    output_lengths: bp (p is the id dimension, indexed by output_seqids)

    Output: b
    """
    # Gather output_counts using target_seqids as index tensor
    B = target_seqids.shape[0]
    target_index = jnp.stack(
            jnp.broadcast_arrays(jnp.arange(B)[None,:], target_seqids), 
            axis=2)
    gd = jax.lax.GatherDimensionNumbers((), (0,1), (0,1)) 

    # both of these are shape bt
    # they give the output sequence length or query sequence length
    # corresponding to the target token at position bt
    output_lengths = jax.lax.gather(output_lengths, tindex, gd, (1,1)).astype(jnp.float32)
    query_lengths = jax.lax.gather(query_lengths, tindex, gd, (1,1)).astype(jnp.float32)
    target_lengths = jax.lax.gather(target_lengths, tindex, gd, (1,1)).astype(jnp.float32)

    # Compute the masked marginal of the attention matrix
    active = 1 - attn_mask
    marg = jnp.sum(attn_matrix, axis=2, where=active, initial=0) # bht
    marg = jnp.mean(marg, axis=1) # bt

    # Further normalize by query sequence length
    # TODO: How to avoid NaNs here?
    have_query = jnp.greater(query_lengths, 0)
    entropy_coeffs = jnp.divide(
            jnp.where(have_query, marg, 1.0),
            jnp.where(have_query, query_lengths, 1.0))

    # Scale each discrete entropy argument so final entropy is in [0, 1]
    # Finally, weight each entropy by output_lengths.  This represents the number of
    # loss terms dependent on this entropy term
    entropy_args = (jnp.log2(entropy_coeffs) 
            * -jnp.log2(target_lengths) 
            * output_lengths)

    final = jnp.sum(entropy_coeffs * entropy_args)
    return final


"""
def _softmax(
    x,
    axis: Optional[Union[int, tuple[int, ...]]] = -1,
    where: Optional[Array] = None,
    initial: Optional[Array] = None) -> Array:
  x_max = jnp.max(x, axis, where=where, initial=initial, keepdims=True)
  unnormalized = jnp.exp(x - x_max)
  result = unnormalized / jnp.sum(unnormalized, axis, where=where, keepdims=True)
  if where is not None:
    result = jnp.where(where, result, 0)
  return result
"""

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
    # print(f'{out.shape=}')
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

def beam_search_step(eos_id, alpha, beta, beam_size, step, enc_mask, logits, dec_kvcache,
        xattn, live_seqs, live_scores, fin_seqs, fin_scores):
    """
    Consumes the next prediction logits
    logits:      bev   (logits for generating tokens at position step+1)

    Returns
    dec_kvcache: lbehsqd  (populated in [:step]
    xattn:  bet   sum of cross-attention coefficients
    live_seqs:   beq,  the set L (live sequences) 
    live_scores: be,  scores for live_seqs (-inf for non-existent)
    fin_seqs:    boq,  the set F (finished sequences)
    fin_scores:  bo,   scores (complete) for fin_seqs (-inf for non-existent)
    """
    score_fn = functools.partial(beam_search_score, alpha, beta, enc_mask)
    # jax.debug.print('step {}, xattn: {}', step, xattn)

    V = logits.shape[2]
    B,E,Q = live_seqs.shape
    new_scores = live_scores[:,:,None] + jax.nn.log_softmax(logits, axis=2)
    # k_prod_inds values index into the e,v space
    live_scores, k_prod_inds = jax.lax.top_k(new_scores.reshape(B,E*V), E) # bk
    # jax.debug.print('step: {}, new_scores[0,0]:\n{}', step, new_scores[0,0])
    # jax.debug.print('step: {}, live_scores[0]:\n{}', step, live_scores[0])
    k_seq_inds, k_tok_inds = jnp.divmod(k_prod_inds, V)
    # Could skip these gather steps if k_seq_inds is the same set as arange(E)
    live_seqs = gather_seqs(live_seqs, k_seq_inds)
    xattn = gather_seqs(xattn, k_seq_inds)
    dec_kvcache = gather_cache(dec_kvcache, k_seq_inds)
    live_seqs = live_seqs.at[:,:,step].set(k_tok_inds)
    # jax.debug.print('step: {}, live_seqs:\n{}', step, live_seqs)

    any_finished = jnp.any(jnp.equal(k_tok_inds, eos_id))
    # jax.debug.print('step: {}, any_finished: {}', step, any_finished)

    def update_fn(args):
        dec_kvcache, xattn, live_seqs, live_scores, fin_seqs, fin_scores = args
        tmp_scores = score_fn(step, live_scores, xattn)
        tmp_scores = jnp.where(live_seqs[:,:,step] == eos_id, tmp_scores, -jnp.inf)
        fin_scores = jnp.concatenate((fin_scores, tmp_scores), axis=1)
        fin_scores, fin_inds = jax.lax.top_k(fin_scores, beam_size)
        fin_seqs = jnp.concatenate((fin_seqs, live_seqs), axis=1)
        # jax.debug.print('Before gather: step: {}, fin_seqs:\n{}', step, fin_seqs)
        fin_seqs = gather_seqs(fin_seqs, fin_inds)
        live_scores = jnp.where(live_seqs[:,:,step] != eos_id, live_scores, -jnp.inf)
        # jax.debug.print('After gather: step: {}, fin_seqs:\n{}', step, fin_seqs)
        return dec_kvcache, xattn, live_seqs, live_scores, fin_seqs, fin_scores

    def passthru_fn(args):
        return args

    args = dec_kvcache, xattn, live_seqs, live_scores, fin_seqs, fin_scores
    return jax.lax.cond(any_finished, update_fn, passthru_fn, args)

def beam_search_score(alpha, beta, enc_mask, out_len, scores, xattn):
    """
    Inputs:
    alpha:    float parameter
    beta:     float parameter
    out_len:  integer parameter
    scores:   bo  log(P(output|input))
    xattn:  bot  (this is sum_j { p_{ij} } in eq 14, where i := t
    enc_mask: bt  positions in target space to ignore

    Scoring function used by beam search, see eq 14 from
    https://arxiv.org/pdf/1609.08144.pdf, as referenced by original AIAYN paper.

    Author's note:
    When alpha = 0 and beta = 0, decoder falls back to pure beam search by probability
    """
    # print(f'beam_search_score: {alpha=}, {beta=}')
    numer = (5.0 + out_len) ** alpha
    denom = 6.0 ** alpha
    lp = numer / denom
    active = 1 - enc_mask
    sum_log_attn = jnp.sum(jnp.log(jnp.minimum(xattn, 1.0)), axis=2, initial=0.0, where=active)
    # sum_log_attn = jnp.log(jnp.minimum(xattn, 1.0)).sum(axis=2)
    # jax.debug.print('sum_log_attn:\n{}', sum_log_attn)
    cp = beta * sum_log_attn 
    return scores / lp + cp


def input_attn(score_model, params, enc_embed, enc_seqids, dec_seqs):
    """
    Compute magnitude of influence each input position has on the probability
    assigned to the decoder sequences

    score_model: create from model.make_score_model
    enc_embed: btm (as output from encoder.embed_layer)
    enc_seqids: bt (ids of each sequence, or -1 if masked
    dec_seqs: bq 

    Returns:
    norms of gradients of embedding vectors w.r.t. log prob
    """

    B, _ = enc_seqids.shape

    primal, vjp_fn = jax.vjp(score_model.apply, params, enc_embed, enc_seqids, dec_seqs)
    out_grads = jnp.ones((B,), dtype=jnp.float32)
    _, _, enc_embed_grad, _, _ = vjp_fn(out_grads)
    return jnp.sqrt(jnp.power(enc_embed_grad, 2).sum(axis=2))


