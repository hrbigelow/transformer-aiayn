def beam_search_score(alpha, beta, x, y, p, log_prob):
    """
    Scoring function used by beam search, see eq 14 from
    https://arxiv.org/pdf/1609.08144.pdf, as referenced by original AIAYN paper.
    
    I: length in tokens
    J: length in tokens
    x: I (token sequence for input sentence)
    y: J (token sequence for output sentence)
    p: shape I,J.  p[i,j] is attention probability of j'th target word on i'th source word
    log_prob: log(P(Y|X)), the log probability assigned by the model

    Author's note:
    "When alpha = 0 and beta = 0, our decoder falls back to pure beam search by probability"
    """
    def lp(y):
        numer = (5.0 + y.shape[0]) ** alpha
        denom = 6.0 ** alpha
        return numer / denom

    def cp(x, y):
        term = t.log(t.min(p.sum(axis=1), 1.0))
        return beta * term.sum(axis=0)

    return log_prob / lp(y) + cp(x, y)


def beam_search(model, x):
    """
    model(x, y) -> T probabilities
    x: input token sequence

    Also, assume model.attn_probs(x, y) -> p
    """
    pass



