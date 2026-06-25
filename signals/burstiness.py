import math
import re

# Sentence length std dev at this value maps to a score of 0.0 (maximally human-like).
_NORMALIZATION_CEILING = 20.0


def compute_burstiness_score(text: str) -> float:
    """Returns float in [0.0, 1.0]: 1.0 = uniform sentences (AI-like), 0.0 = varied (human-like)."""
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    if len(sentences) < 2:
        return 0.5
    lengths = [len(s.split()) for s in sentences]
    mean = sum(lengths) / len(lengths)
    variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    std_dev = math.sqrt(variance)
    return round(max(0.0, 1.0 - (std_dev / _NORMALIZATION_CEILING)), 4)
