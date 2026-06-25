import re

# Single-word markers statistically over-represented in RLHF-tuned LLM output.
_WORD_MARKERS = {
    "delve", "delves", "delving", "delved",
    "testament",
    "furthermore",
    "meticulously", "meticulous",
    "underscore", "underscores", "underscored", "underscoring",
    "nuanced", "nuance",
    "paradigm", "paradigms",
    "leverage", "leveraging", "leveraged", "leverages",
    "pivotal",
    "synergy", "synergies", "synergistic",
    "stakeholders", "stakeholder",
    "robust",
    "streamline", "streamlined", "streamlining",
    "holistic",
    "comprehensive",
    "multifaceted",
    "notably",
    "crucially",
    "groundbreaking",
    "transformative",
    "actionable",
    "innovative",
}

# Multi-word phrase markers checked as substrings in lowercased text.
_PHRASE_MARKERS = [
    "it is important to note",
    "it is worth noting",
    "it is essential to",
    "it's worth noting",
    "in today's",
    "in conclusion",
    "in summary",
    "as mentioned",
    "needless to say",
    "testament to",
    "nuanced approach",
    "multi-faceted",
]

# Occurrences per 100 tokens at which the score saturates to 1.0.
_SATURATION_POINT = 4.0


def compute_ngram_score(text: str) -> float:
    """Returns float in [0.0, 1.0]: 1.0 = dense AI marker phrases (AI-like), 0.0 = clean (human-like)."""
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if not words:
        return 0.0

    word_hits = sum(1 for w in words if w in _WORD_MARKERS)
    phrase_hits = sum(text.lower().count(p) for p in _PHRASE_MARKERS)
    density = ((word_hits + phrase_hits) / len(words)) * 100

    return round(min(1.0, density / _SATURATION_POINT), 4)
