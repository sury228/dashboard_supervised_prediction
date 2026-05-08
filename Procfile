web: gunicorn wsgi:app --workers=4 --worker-class=sync --timeout=120 --bind=0.0.0.0:$PORT
