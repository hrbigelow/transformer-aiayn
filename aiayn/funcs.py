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

def entropy(p, axis, where=None):
    """
    Compute entropy in bits along axis
    p: any shape including at least `axis`
    where: same shape as p
    """
    if where is None:
        return jnp.sum(p * -jnp.log2(p), axis=axis) 
    else:
        assert p.shape == where.shape
        wp0 = jnp.where(where, p, 0.0)
        wp1 = jnp.where(where, p, 1.0)
        return jnp.sum(wp0 * -jnp.log2(wp1), axis=axis)

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

def cross_entropy(q, p_logits, axis):
    p_logits = jax.nn.log_softmax(p_logits, axis)
    log2e = jnp.log2(jnp.exp(1.0))
    xent = jnp.sum(safe_xy(q, - p_logits), axis)
    return xent * log2e

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
    # print(f'beam_search_score: {alpha=}, {beta=}')
    numer = (5.0 + out_len) ** alpha
    denom = 6.0 ** alpha
    lp = numer / denom
    sum_log_attn = jnp.log(jnp.minimum(xattn, 1.0)).sum(axis=2)
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


