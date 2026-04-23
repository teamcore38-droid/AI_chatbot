import threading

from flask import Flask, jsonify, render_template, request

from inference_engine import InferenceEngine


app = Flask(__name__)
engine = InferenceEngine()
engine_lock = threading.Lock()


QUICK_ACTIONS = [
    ("Courses", "What courses do you offer?"),
    ("Fees", "Tell me about the fees"),
    ("Requirements", "Tell me about the requirements"),
    ("Duration", "Tell me about the durations"),
    ("Intake", "Tell me about the intake dates"),
    ("Recommend", "Recommend a course for coding"),
    ("Contact", "How can I contact the counselor?"),
]

WELCOME_MESSAGE = (
    "Hello! I can help you explore courses, fees, duration, requirements, intake dates, "
    "contact details, and recommendations based on your interests."
)


def _run_engine(method, *args, **kwargs):
    with engine_lock:
        return method(*args, **kwargs)


@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/chat")
def chat():
    return render_template(
        "chat.html",
        quick_actions=QUICK_ACTIONS,
        welcome_message=WELCOME_MESSAGE,
    )


@app.route("/api/chat", methods=["POST"])
def api_chat():
    payload = request.get_json(silent=True) or {}
    message = (payload.get("message") or "").strip()

    if not message:
        return jsonify({"ok": False, "error": "Message is required."}), 400

    result = _run_engine(engine.get_response, message)
    return jsonify(
        {
            "ok": True,
            "response": result.response,
            "intent": result.intent,
            "source": result.source,
            "confidence": result.confidence,
            "teachable": result.teachable,
        }
    )


@app.route("/api/teach", methods=["POST"])
def api_teach():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    intent = (payload.get("intent") or "").strip().lower()
    answer = (payload.get("answer") or "").strip()

    if not question or not intent or not answer:
        return jsonify({"ok": False, "error": "Question, intent, and answer are required."}), 400

    success = _run_engine(engine.teach_example, question, intent, answer)
    if not success:
        return jsonify({"ok": False, "error": "Could not save the example."}), 400

    return jsonify(
        {
            "ok": True,
            "message": "New knowledge saved successfully.",
        }
    )


@app.route("/health")
def health():
    return jsonify({"ok": True})
