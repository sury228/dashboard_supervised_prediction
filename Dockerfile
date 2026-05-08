FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create necessary directories
RUN mkdir -p uploads models logs

# Environment variables
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:5000/', timeout=5)" || exit 1

# Run with gunicorn
CMD ["gunicorn", "wsgi:app", \
     "--workers=4", \
     "--worker-class=sync", \
     "--threads=2", \
     "--worker-connections=1000", \
     "--max-requests=1000", \
     "--timeout=120", \
     "--bind=0.0.0.0:5000", \
     "--access-logfile=-", \
     "--error-logfile=-"]
