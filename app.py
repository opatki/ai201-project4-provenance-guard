import uuid

from dotenv import load_dotenv

load_dotenv()

from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from audit_log import get_entry, get_log, write_log_entry
from labels import generate_label
from signals.burstiness import compute_burstiness_score
from signals.ngram import compute_ngram_score
from signals.perplexity import compute_perplexity_score
from signals.scoring import compute_confidence_score
from signals.vocabulary import compute_vocabulary_score

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
@limiter.limit("10 per minute")
def submit():
    data = request.get_json(silent=True)
    if not data or "text" not in data or "creator_id" not in data:
        return jsonify({"error": "Missing required fields: text, creator_id"}), 400

    text = data["text"]
    creator_id = data["creator_id"]
    content_id = str(uuid.uuid4())

    perplexity_score = compute_perplexity_score(text)
    burstiness_score = compute_burstiness_score(text)
    vocabulary_score = compute_vocabulary_score(text)
    ngram_score      = compute_ngram_score(text)
    confidence_score = compute_confidence_score(
        perplexity_score, burstiness_score, vocabulary_score, ngram_score
    )
    attribution = _score_to_attribution(confidence_score)
    label       = generate_label(confidence_score)

    write_log_entry({
        "content_id": content_id,
        "creator_id": creator_id,
        "attribution": attribution,
        "confidence": confidence_score,
        "signals": {
            "perplexity": perplexity_score,
            "burstiness": burstiness_score,
            "vocabulary": vocabulary_score,
            "ngram": ngram_score,
        },
        "label": label["classification"],
        "status": "classified",
    })

    return jsonify({
        "content_id": content_id,
        "signals": {
            "perplexity": perplexity_score,
            "burstiness": burstiness_score,
            "vocabulary": vocabulary_score,
            "ngram": ngram_score,
        },
        "confidence_score": confidence_score,
        "attribution": attribution,
        "label": label,
    })


@app.route("/appeal", methods=["POST"])
def appeal():
    data = request.get_json(silent=True)
    if not data or "content_id" not in data or "creator_reasoning" not in data:
        return jsonify({"error": "Missing required fields: content_id, creator_reasoning"}), 400

    content_id         = data["content_id"]
    creator_reasoning  = data["creator_reasoning"]
    creator_id         = data.get("creator_id", "")

    if not get_entry(content_id):
        return jsonify({"error": "content_id not found"}), 404

    appeal_id = str(uuid.uuid4())

    write_log_entry({
        "type": "appeal",
        "appeal_id": appeal_id,
        "content_id": content_id,
        "creator_id": creator_id,
        "appeal_reasoning": creator_reasoning,
        "status": "under_review",
    })

    return jsonify({
        "appeal_id": appeal_id,
        "content_id": content_id,
        "status": "under_review",
        "message": "Appeal received. Your submission is now under review.",
    })


@app.route("/log", methods=["GET"])
def log():
    return jsonify({"entries": get_log()})


if __name__ == "__main__":
    app.run(debug=True)
