import heapq

def beam_search_score(alpha, beta, out_len, log_prob, in_out_attn):
    """
    Scoring function used by beam search, see eq 14 from
    https://arxiv.org/pdf/1609.08144.pdf, as referenced by original AIAYN paper.
    
    in_out_attn: shape I,J.  p[i,j] is attention probability of j'th output token on 
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

    def cp(in_out_attention):
        # the function scoring the input-to-output attention matrix (p, in the paper)
        term = t.log(t.min(in_out_attention.sum(axis=1), 1.0))
        return beta * term.sum(axis=0)

    return log_prob / lp(out_len) + cp(in_out_attn)

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

def beam_search(model, beam_width, length_penalty, x):
    """
    model(x, y) -> T probabilities
    x: input token sequence
    beam_width: max number of states stored at any given time
    a `state` is a single path ending on a node of the T-ary tree 

    Also, assume model.attn_probs(x, y) -> p
    """
    # the set of complete sentences (ending with eos_token), 
    # paired with beam_search_score
    complete = [(0, None) for _ in range(beam_width)]

    # current set of partial sentences (not ending with eos_token)
    # paired with log_prob score
    partial = [(0, None) for _ in range(beam_width)]

    begin_tok = 0
    dq = [(begin_tok, 0)]  # (token, depth)
    path = []
    while dq:
        node, depth = dq.pop(0)
        if depth == len(path):
            path.append(node)
        path[depth] = node
        for nbor in neighbors(n):
            if PASSED(heap, path, nbor): # score path + nbor
                dq.append((nbor, depth+1))




