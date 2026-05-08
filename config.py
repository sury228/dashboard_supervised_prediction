"""
Flask application configuration for different environments.
"""

import os
from datetime import timedelta

class Config:
    """Base configuration."""
    # Flask
    SECRET_KEY = os.environ.get("SECRET_KEY", "change-me-in-production")
    TESTING = False
    DEBUG = False
    
    # Upload
    UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "uploads")
    MODEL_FOLDER = os.environ.get("MODEL_FOLDER", "models")
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_FILE_SIZE_MB", 50)) * 1024 * 1024
    ALLOWED_EXTENSIONS = {"csv"}
    
    # Session
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Logging
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # ML Engine
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5


class DevelopmentConfig(Config):
    """Development environment configuration."""
    DEBUG = True
    TESTING = False
    SESSION_COOKIE_SECURE = False
    LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    """Production environment configuration."""
    DEBUG = False
    TESTING = False
    SESSION_COOKIE_SECURE = True
    
    # Ensure SECRET_KEY is set in production
    @classmethod
    def __init__(cls):
        if os.environ.get("SECRET_KEY") == "change-me-in-production":
            raise ValueError(
                "SECRET_KEY environment variable not set! "
                "Set a secure random key in production."
            )


class TestingConfig(Config):
    """Testing environment configuration."""
    TESTING = True
    DEBUG = True
    UPLOAD_FOLDER = "test_uploads"
    MODEL_FOLDER = "test_models"
    SESSION_COOKIE_SECURE = False


# Configuration dictionary
config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig,
}


def get_config():
    """Get configuration based on FLASK_ENV."""
    env = os.environ.get("FLASK_ENV", "development")
    return config.get(env, config["default"])
