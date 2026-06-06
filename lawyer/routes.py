"""
Lawyer blueprint — document upload, management, and dashboard.
Only accessible to verified lawyers (role='lawyer') or admins.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from flask import (Blueprint, render_template, request, redirect,
                   url_for, flash, session, jsonify)

from auth.db import (
    get_user_documents, save_document, update_document_status,
    get_document, delete_document, qdrant_collection_for_user,
)
from auth.decorators import lawyer_required, get_current_user
from lawyer.processor import process_document, delete_document_vectors

lawyer_bp = Blueprint("lawyer_bp", __name__, url_prefix="/lawyer",
                      template_folder="../templates/lawyer")

UPLOAD_DIR = Path("data/lawyer_uploads/documents")
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@lawyer_bp.route("/")
@lawyer_required
def dashboard():
    user = get_current_user()
    documents = get_user_documents(user["id"])
    collection = qdrant_collection_for_user(user["id"])

    total_chunks = sum(d["chunk_count"] for d in documents
                       if d["processing_status"] == "completed")
    return render_template("lawyer/dashboard.html",
                           user=user, documents=documents,
                           collection=collection, total_chunks=total_chunks)


@lawyer_bp.route("/upload", methods=["GET", "POST"])
@lawyer_required
def upload():
    user = get_current_user()

    if request.method == "GET":
        return render_template("lawyer/upload.html", user=user)

    if "document" not in request.files:
        flash("لم يتم اختيار ملف", "error")
        return redirect(url_for("lawyer_bp.upload"))

    f = request.files["document"]
    if not f.filename:
        flash("لم يتم اختيار ملف", "error")
        return redirect(url_for("lawyer_bp.upload"))

    if not _allowed_file(f.filename):
        flash("صيغة الملف غير مدعومة. الصيغ المدعومة: PDF, DOCX, TXT", "error")
        return redirect(url_for("lawyer_bp.upload"))

    # Read file content to check size
    content = f.read()
    if len(content) > MAX_FILE_SIZE:
        flash("حجم الملف يتجاوز 10 ميغابايت", "error")
        return redirect(url_for("lawyer_bp.upload"))
    f.seek(0)

    # Save file
    ext = f.filename.rsplit(".", 1)[1].lower()
    safe_name = f"{uuid.uuid4().hex}.{ext}"
    user_dir = UPLOAD_DIR / str(user["id"])
    user_dir.mkdir(parents=True, exist_ok=True)
    save_path = user_dir / safe_name
    f.save(str(save_path))

    # Save to database
    doc_id = save_document(
        user_id=user["id"],
        filename=safe_name,
        original_filename=f.filename,
        file_type=ext,
        file_size_bytes=len(content),
    )

    # Process immediately (in production: use a task queue)
    collection = qdrant_collection_for_user(user["id"])
    try:
        update_document_status(doc_id, "processing")
        chunk_count = process_document(
            file_path=str(save_path),
            file_type=ext,
            user_id=user["id"],
            doc_id=doc_id,
            collection_name=collection,
        )
        update_document_status(doc_id, "completed", chunk_count=chunk_count)
        flash(f"تم معالجة الملف بنجاح! تم إنشاء {chunk_count} مقطع نصي.", "success")
    except Exception as e:
        update_document_status(doc_id, "failed", error_message=str(e))
        flash(f"فشل في معالجة الملف: {e}", "error")

    return redirect(url_for("lawyer_bp.dashboard"))


@lawyer_bp.route("/delete/<int:doc_id>", methods=["POST"])
@lawyer_required
def delete_doc(doc_id: int):
    user = get_current_user()
    doc = get_document(doc_id)

    if doc is None or doc["user_id"] != user["id"]:
        flash("الملف غير موجود", "error")
        return redirect(url_for("lawyer_bp.dashboard"))

    # Delete vectors from Qdrant
    delete_document_vectors(doc_id, doc["qdrant_collection"])

    # Delete physical file
    file_path = UPLOAD_DIR / str(user["id"]) / doc["filename"]
    if file_path.exists():
        file_path.unlink()

    # Delete from database
    delete_document(doc_id)

    flash("تم حذف الملف بنجاح", "success")
    return redirect(url_for("lawyer_bp.dashboard"))
