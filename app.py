"""
ML Dashboard - Main Flask Application
A complete end-to-end machine learning dashboard for classification and regression tasks.
Production-ready with proper logging, error handling, and configuration management.
"""

import os
import sys
import logging
import traceback
from logging.handlers import RotatingFileHandler

import numpy as np
import pandas as pd
from flask import (Flask, render_template, request, jsonify,
                   send_file, session, redirect, url_for, flash)
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from ml_engine import MLEngine
from config import get_config

# Load environment variables
load_dotenv()


def create_app():
    """Application factory function."""
    # Get configuration
    config = get_config()
    
    # Create Flask app
    app = Flask(__name__)
    app.config.from_object(config)
    
    # Create required directories
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["MODEL_FOLDER"], exist_ok=True)
    
    # Setup logging
    setup_logging(app)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Register routes
    register_routes(app)
    
    # Add security headers
    @app.after_request
    def add_security_headers(response):
        """Add security headers to all responses."""
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'SAMEORIGIN'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        return response
    
    return app


# ─────────────────────────────────────────────
# Logging Configuration
# ─────────────────────────────────────────────
def setup_logging(app):
    """Setup logging for the application."""
    if not app.debug and not app.testing:
        # File logging for production
        if not os.path.exists("logs"):
            os.mkdir("logs")
        
        file_handler = RotatingFileHandler(
            "logs/ml_dashboard.log",
            maxBytes=10485760,  # 10MB
            backupCount=10
        )
        file_formatter = logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
    
    # Console logging
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(app.config["LOG_FORMAT"])
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(app.config["LOG_LEVEL"])
    app.logger.addHandler(console_handler)
    app.logger.setLevel(app.config["LOG_LEVEL"])
    app.logger.info("ML Dashboard started")


def register_error_handlers(app):
    """Register error handlers for the application."""
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Resource not found"}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f"Internal error: {error}\n{traceback.format_exc()}")
        return jsonify({"error": "Internal server error"}), 500
    
    @app.errorhandler(413)
    def request_entity_too_large(error):
        return jsonify({"error": "File too large"}), 413


def register_routes(app):
    """Register all routes."""
    # Global ML engine instance
    app.ml_engine = MLEngine()
    
    # Helper utilities
    def allowed_file(filename: str) -> bool:
        return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    
    def _ensure_engine_ready():
        """Restore ml_engine state from session if needed."""
        if app.ml_engine.df is None and session.get("dataset_path"):
            path = session["dataset_path"]
            if os.path.exists(path):
                try:
                    app.ml_engine.load_dataset(path)
                    target = session.get("target")
                    features = session.get("features")
                    if target and features:
                        app.ml_engine.configure(target, features)
                except Exception as e:
                    app.logger.error(f"Error restoring engine state: {e}")
    
    # ─────────────────────────────────────────────────────────────────
    # Routes
    # ─────────────────────────────────────────────────────────────────
    
    @app.route("/")
    def index():
        """Home page."""
        return render_template("index.html")
    
    
    @app.route("/upload", methods=["GET", "POST"])
    def upload():
        """Dataset upload page."""
        if request.method == "POST":
            try:
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
                
                app.logger.info(f"File uploaded: {filename}")
                
                try:
                    info = app.ml_engine.load_dataset(filepath)
                    session["dataset_path"] = filepath
                    return render_template("upload.html", info=info)
                except Exception as e:
                    app.logger.error(f"Error reading dataset: {e}")
                    flash(f"Error reading dataset: {e}", "danger")
                    return redirect(request.url)
            except Exception as e:
                app.logger.error(f"Upload error: {e}\n{traceback.format_exc()}")
                flash("An error occurred during upload.", "danger")
                return redirect(request.url)
        
        return render_template("upload.html", info=None)
    
    
    @app.route("/configure", methods=["POST"])
    def configure():
        """Receive target + feature selection, return problem-type."""
        _ensure_engine_ready()
        try:
            data = request.get_json()
            target   = data.get("target")
            features = data.get("features", [])
            
            if not target or not features:
                return jsonify({"error": "Target and features must be selected."}), 400
            
            prob_type = app.ml_engine.configure(target, features)
            session["target"]       = target
            session["features"]     = features
            session["problem_type"] = prob_type
            
            app.logger.info(f"Configuration: target={target}, features={len(features)}, type={prob_type}")
            return jsonify({"problem_type": prob_type})
        except Exception as e:
            app.logger.error(f"Configuration error: {e}")
            return jsonify({"error": str(e)}), 500
    
    
    @app.route("/model-selection")
    def model_selection():
        """Model selection page (problem type already detected)."""
        prob_type = session.get("problem_type")
        if not prob_type:
            flash("Please upload and configure a dataset first.", "warning")
            return redirect(url_for("upload"))
        return render_template("model_selection.html", problem_type=prob_type)
    
    
    @app.route("/train", methods=["POST"])
    def train():
        """Train selected models and return results as JSON."""
        _ensure_engine_ready()
        try:
            data           = request.get_json()
            selected_models = data.get("models", [])
            tune           = data.get("tune", False)
            
            if not selected_models:
                return jsonify({"error": "No models selected."}), 400
            
            app.logger.info(f"Training {len(selected_models)} models with tuning={tune}")
            results = app.ml_engine.train_models(selected_models, tune=tune)
            
            # Save best model
            best_model_name = results["best_model"]
            model_path = os.path.join(app.config["MODEL_FOLDER"], "best_model.pkl")
            app.ml_engine.save_model(model_path)
            session["model_path"] = model_path
            session["best_model"] = best_model_name
            
            app.logger.info(f"Training complete. Best model: {best_model_name}")
            return jsonify(results)
        except Exception as e:
            app.logger.error(f"Training error: {e}\n{traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500
    
    
    @app.route("/results")
    def results():
        """Results dashboard page."""
        if not session.get("best_model"):
            flash("No trained model found. Please train first.", "warning")
            return redirect(url_for("model_selection"))
        return render_template("results.html",
                               best_model=session.get("best_model"),
                               problem_type=session.get("problem_type"))
    
    
    @app.route("/get-results")
    def get_results():
        """Return cached training results as JSON."""
        _ensure_engine_ready()
        try:
            results = app.ml_engine.get_last_results()
            return jsonify(results)
        except Exception as e:
            app.logger.error(f"Get results error: {e}")
            return jsonify({"error": str(e)}), 500
    
    
    @app.route("/predict", methods=["POST"])
    def predict():
        """Run inference on user-supplied input values."""
        _ensure_engine_ready()
        try:
            data = request.get_json()
            input_values = data.get("values", {})
            
            prediction, confidence = app.ml_engine.predict(input_values)
            app.logger.debug(f"Prediction made with confidence: {confidence}")
            return jsonify({"prediction": prediction, "confidence": confidence})
        except Exception as e:
            app.logger.error(f"Prediction error: {e}")
            return jsonify({"error": str(e)}), 500
    
    
    @app.route("/download-model")
    def download_model():
        """Send the saved best-model pickle to the browser."""
        try:
            model_path = session.get("model_path",
                                     os.path.join(app.config["MODEL_FOLDER"], "best_model.pkl"))
            if not os.path.exists(model_path):
                flash("No trained model available to download.", "warning")
                return redirect(url_for("results"))
            
            app.logger.info(f"Model downloaded: {model_path}")
            return send_file(model_path, as_attachment=True, download_name="best_model.pkl")
        except Exception as e:
            app.logger.error(f"Download error: {e}")
            flash("Error downloading model.", "danger")
            return redirect(url_for("results"))
    
    
    @app.route("/get-plots")
    def get_plots():
        """Generate and return plots as base-64 PNGs."""
        _ensure_engine_ready()
        try:
            plots = app.ml_engine.generate_plots()
            return jsonify(plots)
        except Exception as e:
            app.logger.error(f"Plot generation error: {e}")
            return jsonify({"error": str(e)}), 500
    
    
    @app.route("/get-dataset-info")
    def get_dataset_info():
        """Return stored dataset info (shape, missing values, preview)."""
        _ensure_engine_ready()
        try:
            info = app.ml_engine.get_dataset_info()
            return jsonify(info)
        except Exception as e:
            app.logger.error(f"Dataset info error: {e}")
            return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app = create_app()
    # Use environment port if set, otherwise default to 5000
    port = int(os.environ.get("PORT", 5000))
    # In production, this will be run with gunicorn instead
    app.run(host="0.0.0.0", port=port, debug=False)
