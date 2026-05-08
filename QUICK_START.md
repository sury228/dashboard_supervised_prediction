# ML Dashboard - Quick Start Guide

## 🚀 Quick Deployment

### Option 1: Local Development (Easiest)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run app
python app.py

# 3. Open browser
Visit http://localhost:5000
```

### Option 2: Docker (Recommended for consistency)
```bash
# 1. Build and run
docker-compose up

# 2. Open browser
Visit http://localhost:5000

# 3. Stop
docker-compose down
```

### Option 3: Heroku (Free option - limited)
```bash
# 1. Install Heroku CLI
# 2. Login
heroku login

# 3. Create app
heroku create your-app-name

# 4. Set environment
heroku config:set SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')
heroku config:set FLASK_ENV=production

# 5. Deploy
git push heroku main

# 6. View logs
heroku logs --tail
```

### Option 4: Linux Server with Systemd
```bash
# 1. Clone repository
git clone <repo> /opt/ml-dashboard
cd /opt/ml-dashboard

# 2. Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with production values

# 4. Setup systemd service
sudo cp ml-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ml-dashboard
sudo systemctl start ml-dashboard

# 5. Setup nginx reverse proxy
sudo cp nginx.conf /etc/nginx/sites-available/ml-dashboard
sudo ln -s /etc/nginx/sites-available/ml-dashboard /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# 6. Verify running
curl http://localhost:5000
```

---

## 📋 Environment Setup

### Generate Secure SECRET_KEY
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```
Copy the output and set as `SECRET_KEY` in `.env`

### Create .env file
```bash
cp .env.example .env
nano .env  # Edit with your settings
```

---

## 🔒 Production Security Checklist

- [ ] `SECRET_KEY` is random and unique (never use default)
- [ ] `FLASK_ENV=production` is set
- [ ] Debug mode is OFF
- [ ] HTTPS/SSL certificate installed
- [ ] All environment variables configured
- [ ] Database backups enabled (if using DB)
- [ ] Logs are being written and rotated
- [ ] Monitoring/alerts configured

---

## 📊 Monitoring Logs

```bash
# Docker
docker-compose logs -f ml-dashboard

# Linux systemd
sudo journalctl -u ml-dashboard -f

# Direct file
tail -f logs/ml_dashboard.log
```

---

## 🆘 Troubleshooting

### Port 5000 already in use
```bash
# Linux
lsof -i :5000; kill -9 <PID>

# Windows
netstat -ano | findstr :5000; taskkill /PID <PID> /F
```

### Import errors
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Secret key not set
```
ERROR: SECRET_KEY environment variable not set!
Set a secure random key in production.
```
**Solution**: Set `SECRET_KEY` in `.env` file

### File upload not working
- Check `uploads/` directory exists
- Verify it's writable: `ls -la uploads/`
- Check file size < MAX_FILE_SIZE_MB

### Model training timeout
- Reduce dataset size
- Disable hyperparameter tuning in UI
- Increase timeout in nginx/gunicorn config

---

## 📈 Performance Tips

1. **Run with multiple workers** (4-8 for 2-4 CPU cores)
2. **Use reverse proxy** (nginx/Apache) for static files
3. **Enable compression** (gzip)
4. **Cache static assets** in browser
5. **Monitor memory usage** during model training
6. **Use separate worker processes** for long tasks

---

## 🔧 Configuration

Edit `config.py` to modify default settings:
- Session timeout
- File upload limits
- CORS settings
- Database configuration
- Logging levels

---

## 📚 Documentation

- [Full Deployment Guide](README.md)
- [Deployment Checklist](DEPLOYMENT_CHECKLIST.md)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Gunicorn Documentation](https://docs.gunicorn.org/)
- [Docker Documentation](https://docs.docker.com/)

---

## 📞 Support

For issues:
1. Check logs: `tail -f logs/ml_dashboard.log`
2. Enable DEBUG logging: `LOG_LEVEL=DEBUG` in `.env`
3. Review DEPLOYMENT_CHECKLIST.md
4. Check infrastructure (disk space, memory, CPU)

---

**Last Updated**: 2026-05-08
**Version**: 1.0
