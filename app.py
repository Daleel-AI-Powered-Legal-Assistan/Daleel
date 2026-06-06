"""
خادم Flask — مساعد قانون العمل الأردني
POST /chat   <- { "question": "...", "history": [...] }
GET  /health
GET  /eval/  <- منصة التقييم الجماعي
"""
from __future__ import annotations

import os
from flask import Flask, render_template, request, jsonify, session
from chatbot import get_bot
from eval_gui.routes import eval_bp
from auth.routes import auth_bp
from auth.db import init_db, get_user_by_id
from auth.decorators import get_current_user
from lawyer.routes import lawyer_bp
from admin.routes import admin_bp

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")

# Register blueprints
app.register_blueprint(eval_bp)
app.register_blueprint(auth_bp)
app.register_blueprint(lawyer_bp)
app.register_blueprint(admin_bp)

# Initialize user database on startup
with app.app_context():
    init_db()


@app.context_processor
def inject_user():
    """Make current_user available in all templates."""
    return {"current_user": get_current_user()}


@app.route("/")
def index():
    bot = get_bot()
    return render_template("index.html", llm_available=bot._groq_available)


@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data     = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    history  = data.get("history") or []

    if not question:
        return jsonify({"error": "السؤال فارغ"}), 400

    # Pass user_id for personalized retrieval (lawyers only)
    user_id = None
    user = get_current_user()
    if user and user["role"] in ("lawyer", "admin"):
        user_id = user["id"]

    try:
        result = get_bot().chat(question, history=history, user_id=user_id)
    except Exception as e:
        return jsonify({"error": f"خطأ في الخادم: {e}"}), 500
    if "error" in result:
        return jsonify(result), 500
    return jsonify(result)


@app.route("/health")
def health():
    try:
        bot  = get_bot()
        info = bot.qdrant.get_collection("jordan_labor_law")
        pts  = getattr(info, "points_count", None) or getattr(info, "vectors_count", "?")
        return jsonify({
            "ok":        True,
            "points":    pts,
            "llm_ready": bot._groq_available,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
