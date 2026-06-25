import json
import os

from groq import Groq

_client = None

_SYSTEM_PROMPT = """You are a text analysis engine that measures statistical predictability — \
a proxy for AI-generation likelihood.

Analyze the text and return a JSON object with exactly one field: "perplexity_score" (float 0.0–1.0).

Scoring guide:
  1.0 = maximally predictable: formulaic phrasing, statistically average word choices, \
uniform sentence length, smooth boilerplate transitions → strongly AI-generated.
  0.0 = maximally unpredictable: idiosyncratic diction, variable rhythm, personal voice, \
unexpected word choices → strongly human-written.

Return ONLY valid JSON with no extra keys: {"perplexity_score": <float>}"""


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def compute_perplexity_score(text: str) -> float:
    """Returns float in [0.0, 1.0]: 1.0 = likely AI-generated, 0.0 = likely human."""
    response = _get_client().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    result = json.loads(response.choices[0].message.content)
    score = float(result["perplexity_score"])
    return round(max(0.0, min(1.0, score)), 4)
