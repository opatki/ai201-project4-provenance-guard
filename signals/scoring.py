# Weights from spec — sum to exactly 1.0, no renormalization needed.
_W_PERPLEXITY  = 0.35
_W_BURSTINESS  = 0.25
_W_VOCABULARY  = 0.20
_W_NGRAM       = 0.20


def compute_confidence_score(
    perplexity_score: float,
    burstiness_score: float,
    vocabulary_score: float,
    ngram_score: float,
) -> float:
    """Returns float in [0.0, 1.0]: weighted combination of all four signals."""
    score = (
        _W_PERPLEXITY * perplexity_score
        + _W_BURSTINESS * burstiness_score
        + _W_VOCABULARY * vocabulary_score
        + _W_NGRAM      * ngram_score
    )
    return round(max(0.0, min(1.0, score)), 4)
