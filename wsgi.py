"""
WSGI entry point for production servers (Gunicorn, uWSGI, etc.)
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from app import create_app

# Create the Flask application
app = create_app()

if __name__ == "__main__":
    # This is only used for local testing
    # Use Gunicorn for production: gunicorn wsgi:app
    app.run()
