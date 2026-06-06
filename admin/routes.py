"""
Admin blueprint — lawyer verification management.
"""
from __future__ import annotations

from flask import (Blueprint, render_template, request, redirect,
                   url_for, flash, session)

from auth.db import (
    get_pending_verifications, get_all_verifications,
    approve_verification, reject_verification,
)
from auth.decorators import admin_required, get_current_user

admin_bp = Blueprint("admin_bp", __name__, url_prefix="/admin",
                     template_folder="../templates/admin")


@admin_bp.route("/")
@admin_required
def dashboard():
    user = get_current_user()
    pending = get_pending_verifications()
    all_verifications = get_all_verifications()
    return render_template("admin/dashboard.html",
                           user=user, pending=pending,
                           all_verifications=all_verifications)


@admin_bp.route("/approve/<int:vid>", methods=["POST"])
@admin_required
def approve(vid: int):
    user = get_current_user()
    approve_verification(vid, user["id"])
    flash("تم اعتماد المحامي بنجاح", "success")
    return redirect(url_for("admin_bp.dashboard"))


@admin_bp.route("/reject/<int:vid>", methods=["POST"])
@admin_required
def reject(vid: int):
    user = get_current_user()
    reason = request.form.get("reason", "")
    reject_verification(vid, user["id"], reason)
    flash("تم رفض الطلب", "info")
    return redirect(url_for("admin_bp.dashboard"))
