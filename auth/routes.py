"""
Auth blueprint — register, login, logout, lawyer verification submit.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from flask import (Blueprint, render_template, request, redirect,
                   url_for, flash, session)

import re

from auth.db import (
    create_user, authenticate, email_exists,
    validate_lawyer_id, verify_bar_registration,
    submit_lawyer_verification,
    get_verification_by_user, lawyer_id_taken,
    update_user_role,
)
from auth.decorators import login_required, get_current_user


def _validate_password(password: str) -> list[str]:
    """Password must contain uppercase, lowercase, digit, and special char."""
    errors = []
    if not re.search(r"[A-Z]", password):
        errors.append("كلمة المرور يجب أن تحتوي على حرف كبير (A-Z)")
    if not re.search(r"[a-z]", password):
        errors.append("كلمة المرور يجب أن تحتوي على حرف صغير (a-z)")
    if not re.search(r"\d", password):
        errors.append("كلمة المرور يجب أن تحتوي على رقم (0-9)")
    if not re.search(r"[^A-Za-z0-9]", password):
        errors.append("كلمة المرور يجب أن تحتوي على رمز خاص (!@#$...)")
    return errors

auth_bp = Blueprint("auth", __name__, url_prefix="/auth",
                    template_folder="../templates/auth")

UPLOAD_DIR = Path("data/lawyer_uploads/id_documents")
MAX_ID_DOC_SIZE = 5 * 1024 * 1024  # 5 MB


@auth_bp.route("/lawyer-gate")
def lawyer_gate():
    """Landing page for lawyers — if already logged in, redirect to dashboard."""
    user = get_current_user()
    if user:
        if user["role"] == "lawyer":
            return redirect(url_for("lawyer_bp.dashboard"))
    return render_template("auth/lawyer_gate.html")


@auth_bp.route("/lawyer-register", methods=["GET", "POST"])
def lawyer_register():
    """One-step lawyer registration: account + bar verification → instant lawyer role."""
    if request.method == "GET":
        return render_template("auth/lawyer_register.html")

    email = (request.form.get("email") or "").strip()
    password = request.form.get("password", "")
    confirm = request.form.get("confirm_password", "")
    full_name_ar = (request.form.get("full_name_ar") or "").strip()
    lawyer_id = (request.form.get("lawyer_id") or "").strip()

    errors = []
    if not email or "@" not in email:
        errors.append("بريد إلكتروني غير صالح")
    errors.extend(_validate_password(password))
    if password != confirm:
        errors.append("كلمتا المرور غير متطابقتين")
    if not full_name_ar:
        errors.append("الاسم الكامل بالعربية مطلوب")
    if email_exists(email):
        errors.append("البريد الإلكتروني مسجل مسبقاً")
    if not validate_lawyer_id(lawyer_id):
        errors.append("رقم النقابة يجب أن يكون 6 أرقام بالضبط")
    elif lawyer_id_taken(lawyer_id):
        errors.append("رقم النقابة مسجل مسبقاً في النظام")

    # Verify against licensed lawyers registry
    if not errors:
        bar_error = verify_bar_registration(lawyer_id, full_name_ar)
        if bar_error:
            errors.append(bar_error)

    if errors:
        for e in errors:
            flash(e, "error")
        return render_template("auth/lawyer_register.html",
                               email=email, full_name_ar=full_name_ar,
                               lawyer_id=lawyer_id)

    # Create account with lawyer role directly
    user_id = create_user(email, password, full_name_ar, role="lawyer")
    session["user_id"] = user_id
    flash("تم إنشاء حساب المحامي بنجاح! مرحباً بك في دليل.", "success")
    return redirect(url_for("lawyer_bp.dashboard"))


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("auth/register.html")

    email = (request.form.get("email") or "").strip()
    password = request.form.get("password", "")
    confirm = request.form.get("confirm_password", "")
    full_name = (request.form.get("full_name") or "").strip()

    # Validation
    errors = []
    if not email or "@" not in email:
        errors.append("بريد إلكتروني غير صالح")
    errors.extend(_validate_password(password))
    if password != confirm:
        errors.append("كلمتا المرور غير متطابقتين")
    if not full_name:
        errors.append("الاسم الكامل مطلوب")
    if email_exists(email):
        errors.append("البريد الإلكتروني مسجل مسبقاً")

    if errors:
        for e in errors:
            flash(e, "error")
        return render_template("auth/register.html",
                               email=email, full_name=full_name)

    user_id = create_user(email, password, full_name)
    session["user_id"] = user_id
    flash("تم إنشاء الحساب بنجاح!", "success")
    next_url = request.args.get("next") or url_for("index")
    return redirect(next_url)


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("auth/login.html")

    email = (request.form.get("email") or "").strip()
    password = request.form.get("password", "")

    user = authenticate(email, password)
    if user is None:
        flash("بريد إلكتروني أو كلمة مرور خاطئة", "error")
        return render_template("auth/login.html", email=email)

    session["user_id"] = user["id"]
    flash(f"أهلاً {user['full_name']}!", "success")

    next_url = request.args.get("next") or url_for("index")
    return redirect(next_url)


@auth_bp.route("/logout")
def logout():
    session.clear()
    flash("تم تسجيل الخروج", "success")
    return redirect(url_for("index"))


@auth_bp.route("/verify-lawyer", methods=["GET", "POST"])
@login_required
def verify_lawyer():
    user = get_current_user()
    existing = get_verification_by_user(user["id"])

    if user["role"] == "lawyer":
        flash("أنت محامٍ معتمد بالفعل!", "success")
        return redirect(url_for("lawyer_bp.dashboard"))

    if existing and existing["status"] == "pending":
        flash("طلبك قيد المراجعة بالفعل", "info")
        return render_template("auth/verify_lawyer.html",
                               verification=existing, user=user)

    if request.method == "GET":
        return render_template("auth/verify_lawyer.html",
                               verification=existing, user=user)

    lawyer_id = (request.form.get("lawyer_id") or "").strip()
    full_name_ar = (request.form.get("full_name_ar") or "").strip()

    errors = []
    if not validate_lawyer_id(lawyer_id):
        errors.append("رقم النقابة يجب أن يكون 6 أرقام بالضبط")
    elif lawyer_id_taken(lawyer_id):
        errors.append("رقم النقابة مسجل مسبقاً في النظام")
    if not full_name_ar:
        errors.append("الاسم الكامل بالعربية مطلوب")

    # Verify against licensed lawyers registry
    if not errors:
        bar_error = verify_bar_registration(lawyer_id, full_name_ar)
        if bar_error:
            errors.append(bar_error)

    # Handle ID document upload (optional but recommended)
    id_doc_path = None
    if "id_document" in request.files:
        f = request.files["id_document"]
        if f.filename:
            if f.content_length and f.content_length > MAX_ID_DOC_SIZE:
                errors.append("حجم الملف يتجاوز 5 ميغابايت")
            else:
                ext = Path(f.filename).suffix.lower()
                if ext not in (".jpg", ".jpeg", ".png", ".pdf"):
                    errors.append("صيغة الملف غير مدعومة (JPG, PNG, PDF فقط)")
                else:
                    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                    safe_name = f"{uuid.uuid4().hex}{ext}"
                    save_path = UPLOAD_DIR / safe_name
                    f.save(str(save_path))
                    id_doc_path = str(save_path)

    if errors:
        for e in errors:
            flash(e, "error")
        return render_template("auth/verify_lawyer.html",
                               verification=existing, user=user,
                               lawyer_id=lawyer_id, full_name_ar=full_name_ar)

    submit_lawyer_verification(user["id"], lawyer_id, full_name_ar, id_doc_path)
    flash("تم إرسال طلب التحقق بنجاح! سيتم مراجعته من قبل المسؤول.", "success")
    return redirect(url_for("auth.verify_lawyer"))
