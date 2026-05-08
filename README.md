# ML Dashboard - Deployment Guide

A production-ready machine learning dashboard for classification and regression tasks built with Flask and scikit-learn.

## Features

- **End-to-end ML Pipeline**: Upload data, configure features, train models, and make predictions
- **Multi-model Support**: 
  - Classification: Logistic Regression, Random Forest, SVM, KNN, Gradient Boosting
  - Regression: Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting
- **Hyperparameter Tuning**: GridSearchCV and RandomizedSearchCV support
- **Model Evaluation**: Comprehensive metrics, confusion matrices, and visualizations
- **Model Persistence**: Save and download trained models
- **Production Ready**: Logging, error handling, security headers, environment-based configuration

## Project Structure

```
.
├── app.py                 # Flask application with routes
├── ml_engine.py          # ML pipeline and model logic
├── config.py             # Configuration management
├── wsgi.py              # WSGI entry point for production
├── requirements.txt      # Python dependencies
├── Procfile             # Heroku deployment configuration
├── runtime.txt          # Python version specification
├── .env.example         # Example environment variables
├── .gitignore           # Git ignore file
├── templates/           # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── upload.html
│   ├── model_selection.html
│   └── results.html
├── static/              # CSS, JS, images
│   ├── css/
│   ├── js/
│   └── images/
├── models/              # Trained models (Git ignored)
├── uploads/             # Uploaded datasets (Git ignored)
└── logs/               # Application logs (Git ignored)
```

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd final_project_ml
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
```bash
# Create .env file from .env.example
cp .env.example .env

# Edit .env and set:
# - SECRET_KEY: A secure random string (use Python: import secrets; secrets.token_hex(32))
# - FLASK_ENV: development, production, or testing
# - LOG_LEVEL: DEBUG, INFO, WARNING, ERROR
```

### 5. Run Development Server
```bash
python app.py
```
Visit http://localhost:5000 in your browser.

## Development vs Production

### Development
```bash
export FLASK_ENV=development
python app.py
```
- Debug mode enabled
- Auto-reloader active
- Detailed logging

### Production
```bash
export FLASK_ENV=production
export SECRET_KEY=<your-secure-key>
gunicorn wsgi:app --workers=4 --bind=0.0.0.0:5000
```
- Debug mode disabled
- File logging with rotation
- Security headers enabled
- Session cookies are secure/HTTPOnly

## Deployment

### Heroku

1. **Install Heroku CLI**
   ```bash
   https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Create Heroku App**
   ```bash
   heroku create your-app-name
   ```

3. **Set Environment Variables**
   ```bash
   heroku config:set FLASK_ENV=production
   heroku config:set SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')
   ```

4. **Deploy**
   ```bash
   git push heroku main
   ```

5. **View Logs**
   ```bash
   heroku logs --tail
   ```

### Docker

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_ENV=production
EXPOSE 5000

CMD ["gunicorn", "wsgi:app", "--workers=4", "--bind=0.0.0.0:5000", "--timeout=120"]
```

Build and run:
```bash
docker build -t ml-dashboard .
docker run -p 5000:5000 -e SECRET_KEY=<your-key> ml-dashboard
```

### AWS, Google Cloud, Azure

Use similar approaches with:
- Cloud Functions / App Engine / App Service
- Environment variable configuration
- PostgreSQL for session/data storage (if needed)
- S3 / Google Cloud Storage for model/upload persistence

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_ENV` | development | Environment: development, production, testing |
| `SECRET_KEY` | change-me | Session encryption key - MUST be set in production |
| `PORT` | 5000 | Server port |
| `UPLOAD_FOLDER` | uploads | Dataset upload directory |
| `MODEL_FOLDER` | models | Trained models directory |
| `MAX_FILE_SIZE_MB` | 50 | Maximum upload file size |
| `LOG_LEVEL` | INFO | Logging level |

## Configuration

Edit `config.py` to modify:
- Session timeout
- File upload limits
- Database connections
- Model hyperparameters
- Cross-origin settings

## Logging

Logs are stored in `logs/ml_dashboard.log` with rotation:
- **Max file size**: 10MB
- **Backup count**: 10 files
- **Format**: Timestamp, level, message, location

View logs:
```bash
# Development: Console output
# Production: logs/ml_dashboard.log
tail -f logs/ml_dashboard.log
```

## Security Best Practices

✅ **Implemented:**
- Secure session cookies (HTTPOnly, SameSite)
- Secret key management via environment variables
- Input validation and sanitization (secure_filename)
- CSRF protection via Flask session
- Security headers (X-Content-Type-Options, X-Frame-Options, X-XSS-Protection)
- File upload size limits
- Comprehensive error handling

⚠️ **Additional Recommendations:**
- Use HTTPS in production
- Enable rate limiting for API endpoints
- Add authentication/authorization
- Use PostgreSQL for session storage (vs in-memory)
- Store uploaded files in S3/GCS (vs local filesystem)
- Implement database for audit logging
- Add input validation on frontend and backend
- Regular security updates and dependency scanning

## Troubleshooting

### Port Already in Use
```bash
# Linux/macOS
lsof -i :5000
kill -9 <PID>

# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

### Missing Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Session Issues
Clear browser cookies or:
```bash
# Restart server
# Sessions are stored in memory by default
```

### File Upload Issues
- Check `uploads/` directory exists and is writable
- Verify file size < MAX_FILE_SIZE_MB
- Ensure CSV format is correct

### Model Training Timeout
- Reduce dataset size
- Disable hyperparameter tuning
- Increase server timeout in production settings

## Performance Optimization

1. **Model Caching**: Trained models are cached in memory
2. **Dataset Caching**: Datasets stored in session
3. **Worker Processes**: Run with multiple Gunicorn workers (4-8)
4. **Database**: Consider PostgreSQL for session storage at scale

## Testing

```bash
# Set testing environment
export FLASK_ENV=testing

# Run unit tests (create test_app.py)
pytest test_app.py
```

## Maintenance

- **Dependency Updates**: Regularly run `pip list --outdated`
- **Log Rotation**: Automatic (max 10 x 10MB files)
- **Model Management**: Periodically retrain with new data
- **Database Cleanup**: Remove old sessions and logs

## Support & Documentation

- **Flask**: https://flask.palletsprojects.com/
- **scikit-learn**: https://scikit-learn.org/
- **Gunicorn**: https://gunicorn.org/
- **Heroku**: https://devcenter.heroku.com/

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
