"""
Role-based access decorators for Flask routes.

Uses Flask session to track logged-in user.
"""
from __future__ import annotations

from functools import wraps
from flask import session, redirect, url_for, flash, request
from auth.db import get_user_by_id


def get_current_user() -> dict | None:
    """Return current user dict from session, or None."""
    uid = session.get("user_id")
    if uid is None:
        return None
    return get_user_by_id(uid)


def login_required(f):
    """Redirect to login if not authenticated."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("يرجى تسجيل الدخول أولاً", "warning")
            return redirect(url_for("auth.login", next=request.path))
        return f(*args, **kwargs)
    return wrapper


def lawyer_required(f):
    """Only verified lawyers (role='lawyer') or admins."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        user = get_current_user()
        if user is None:
            flash("يرجى تسجيل الدخول أولاً", "warning")
            return redirect(url_for("auth.login"))
        if user["role"] not in ("lawyer", "admin"):
            flash("هذه الميزة متاحة للمحامين المعتمدين فقط", "error")
            return redirect(url_for("index"))
        return f(*args, **kwargs)
    return wrapper


def admin_required(f):
    """Only admins."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        user = get_current_user()
        if user is None:
            flash("يرجى تسجيل الدخول أولاً", "warning")
            return redirect(url_for("auth.login"))
        if user["role"] != "admin":
            flash("غير مصرح لك بالوصول", "error")
            return redirect(url_for("index"))
        return f(*args, **kwargs)
    return wrapper
