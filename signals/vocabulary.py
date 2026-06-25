import re

_WINDOW = 50       # tokens per sliding window
_LOW_BOUND = 0.55  # MATTR at or below this → score = 1.0 (AI-like)
_HIGH_BOUND = 0.85 # MATTR at or above this → score = 0.0 (human-like)


def _mattr(words: list) -> float:
    """Moving Average Type-Token Ratio over a 50-token sliding window."""
    if len(words) < _WINDOW:
        # Fewer words than one full window — plain TTR
        return len(set(words)) / len(words)
    ttrs = [
        len(set(words[i:i + _WINDOW])) / _WINDOW
        for i in range(len(words) - _WINDOW + 1)
    ]
    return sum(ttrs) / len(ttrs)


def compute_vocabulary_score(text: str) -> float:
    """Returns float in [0.0, 1.0]: 1.0 = low vocab diversity (AI-like), 0.0 = rich diversity (human-like)."""
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if not words:
        return 0.5
    mattr_raw = _mattr(words)
    score = 1.0 - ((mattr_raw - _LOW_BOUND) / (_HIGH_BOUND - _LOW_BOUND))
    return round(max(0.0, min(1.0, score)), 4)
