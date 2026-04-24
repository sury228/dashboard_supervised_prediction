"""
ML Dashboard - Main Flask Application
A complete end-to-end machine learning dashboard for classification and regression tasks.
"""

import os
import json
import base64
import pickle
import traceback

import numpy as np
import pandas as pd
from flask import (Flask, render_template, request, jsonify,
                   send_file, session, redirect, url_for, flash)
from werkzeug.utils import secure_filename

from ml_engine import MLEngine

# ─────────────────────────────────────────────
# App Configuration
# ─────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "ml_dashboard_secret_2024"

UPLOAD_FOLDER   = "uploads"
MODEL_FOLDER    = "models"
ALLOWED_EXTS    = {"csv"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MODEL_FOLDER"]  = MODEL_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB limit

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER,  exist_ok=True)

# Global ML engine instance (kept in process memory between requests)
ml_engine: MLEngine = MLEngine()


def _ensure_engine_ready():
    """Restore ml_engine state from session if the server restarted."""
    if ml_engine.df is None and session.get("dataset_path"):
        path = session["dataset_path"]
        if os.path.exists(path):
            ml_engine.load_dataset(path)
            target = session.get("target")
            features = session.get("features")
            if target and features:
                ml_engine.configure(target, features)


# ─────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/")
def index():
    """Home page."""
    return render_template("index.html")


# ── Upload ────────────────────────────────────
@app.route("/upload", methods=["GET", "POST"])
def upload():
    """Dataset upload page."""
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part in request.", "danger")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("No file selected.", "danger")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Only CSV files are allowed.", "danger")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Load & analyse
        try:
            info = ml_engine.load_dataset(filepath)
            session["dataset_path"] = filepath
            return render_template("upload.html", info=info)
        except Exception as e:
            flash(f"Error reading dataset: {e}", "danger")
            return redirect(request.url)

    return render_template("upload.html", info=None)


# ── Column configuration (AJAX) ───────────────
@app.route("/configure", methods=["POST"])
def configure():
    """Receive target + feature selection, return problem-type."""
    _ensure_engine_ready()
    data = request.get_json()
    target   = data.get("target")
    features = data.get("features", [])

    if not target or not features:
        return jsonify({"error": "Target and features must be selected."}), 400

    try:
        prob_type = ml_engine.configure(target, features)
        session["target"]       = target
        session["features"]     = features
        session["problem_type"] = prob_type
        return jsonify({"problem_type": prob_type})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Model selection ───────────────────────────
@app.route("/model-selection")
def model_selection():
    """Model selection page (problem type already detected)."""
    prob_type = session.get("problem_type")
    if not prob_type:
        flash("Please upload and configure a dataset first.", "warning")
        return redirect(url_for("upload"))
    return render_template("model_selection.html", problem_type=prob_type)


# ── Training (AJAX long-running) ──────────────
@app.route("/train", methods=["POST"])
def train():
    """Train selected models and return results as JSON."""
    _ensure_engine_ready()
    data           = request.get_json()
    selected_models = data.get("models", [])
    tune           = data.get("tune", False)

    if not selected_models:
        return jsonify({"error": "No models selected."}), 400

    try:
        results = ml_engine.train_models(selected_models, tune=tune)

        # Save best model
        best_model_name = results["best_model"]
        model_path = os.path.join(app.config["MODEL_FOLDER"], "best_model.pkl")
        ml_engine.save_model(model_path)
        session["model_path"] = model_path
        session["best_model"] = best_model_name

        return jsonify(results)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Results dashboard ─────────────────────────
@app.route("/results")
def results():
    """Results dashboard page."""
    if not session.get("best_model"):
        flash("No trained model found. Please train first.", "warning")
        return redirect(url_for("model_selection"))
    return render_template("results.html",
                           best_model=session.get("best_model"),
                           problem_type=session.get("problem_type"))


# ── Fetch stored results (AJAX) ───────────────
@app.route("/get-results")
def get_results():
    """Return cached training results as JSON."""
    _ensure_engine_ready()
    try:
        results = ml_engine.get_last_results()
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Prediction ────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """Run inference on user-supplied input values."""
    _ensure_engine_ready()
    data = request.get_json()
    input_values = data.get("values", {})

    try:
        prediction, confidence = ml_engine.predict(input_values)
        return jsonify({"prediction": prediction, "confidence": confidence})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Download best model ───────────────────────
@app.route("/download-model")
def download_model():
    """Send the saved best-model pickle to the browser."""
    model_path = session.get("model_path",
                             os.path.join(app.config["MODEL_FOLDER"], "best_model.pkl"))
    if not os.path.exists(model_path):
        flash("No trained model available to download.", "warning")
        return redirect(url_for("results"))
    return send_file(model_path, as_attachment=True, download_name="best_model.pkl")


# ── Feature importance / plots (AJAX) ────────
@app.route("/get-plots")
def get_plots():
    """Generate and return plots as base-64 PNGs."""
    _ensure_engine_ready()
    try:
        plots = ml_engine.generate_plots()
        return jsonify(plots)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Dataset info (AJAX) ───────────────────────
@app.route("/get-dataset-info")
def get_dataset_info():
    """Return stored dataset info (shape, missing values, preview)."""
    _ensure_engine_ready()
    try:
        info = ml_engine.get_dataset_info()
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
