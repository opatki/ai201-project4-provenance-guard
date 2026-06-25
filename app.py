import uuid

from dotenv import load_dotenv

load_dotenv()

from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from audit_log import write_log_entry, get_log
from signals.perplexity import compute_perplexity_score

app = Flask(__name__)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["100 per minute"],
)


def _score_to_attribution(score: float) -> str:
    if score >= 0.85:
        return "definitely_ai"
    if score >= 0.65:
        return "likely_ai"
    if score >= 0.36:
        return "uncertain"
    if score >= 0.16:
        return "likely_human"
    return "definitely_human"


@app.route("/submit", methods=["POST"])
def submit():
    data = request.get_json(silent=True)
    if not data or "text" not in data or "creator_id" not in data:
        return jsonify({"error": "Missing required fields: text, creator_id"}), 400

    text = data["text"]
    creator_id = data["creator_id"]
    content_id = str(uuid.uuid4())

    perplexity_score = compute_perplexity_score(text)
    attribution = _score_to_attribution(perplexity_score)

    write_log_entry({
        "content_id": content_id,
        "creator_id": creator_id,
        "attribution": attribution,
        "confidence": None,         # M4: computed from all signals
        "llm_score": perplexity_score,
        "status": "classified",
    })

    return jsonify({
        "content_id": content_id,
        "perplexity_score": perplexity_score,
        "attribution": attribution,
        "confidence_score": None,   # M4
        "label": None,              # M5
    })


@app.route("/log", methods=["GET"])
def log():
    return jsonify({"entries": get_log()})


if __name__ == "__main__":
    app.run(debug=True)
