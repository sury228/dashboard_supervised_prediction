# ML Dashboard - Deployment Checklist

## Pre-Deployment (Local Testing)

- [ ] All Python dependencies installed: `pip install -r requirements.txt`
- [ ] No syntax errors: `python -m py_compile app.py ml_engine.py config.py`
- [ ] Application runs locally: `python app.py`
- [ ] Can upload sample CSV file
- [ ] Model training completes successfully
- [ ] Predictions work correctly
- [ ] No hardcoded secrets or credentials in code
- [ ] Git history cleaned of sensitive data
- [ ] Tests pass: `pytest` (if tests exist)

## Environment Setup

- [ ] `.env` file created and configured (never commit to git)
- [ ] `SECRET_KEY` generated: `python -c "import secrets; print(secrets.token_hex(32))"`
- [ ] `FLASK_ENV` set to `production`
- [ ] `LOG_LEVEL` appropriate for environment
- [ ] File upload directory writable
- [ ] Model storage directory writable
- [ ] Logs directory created and writable

## Security Review

- [ ] No debug mode enabled in production
- [ ] Session cookies are secure (HTTPOnly, SameSite)
- [ ] CORS properly configured if needed
- [ ] Input validation on all endpoints
- [ ] File upload validation (type, size)
- [ ] SQL injection prevention (using ORM/parameterized queries if DB used)
- [ ] XSS protection (Jinja2 auto-escaping enabled)
- [ ] CSRF protection enabled
- [ ] Rate limiting considered
- [ ] Error messages don't expose sensitive information

## Database (if applicable)

- [ ] Database migrations run: `flask db upgrade`
- [ ] Database backups configured
- [ ] Connection pooling configured
- [ ] Read replicas considered for load balancing

## Infrastructure

- [ ] Server/VM provisioned with adequate resources
- [ ] Python 3.11+ installed
- [ ] Firewall rules configured (allow 443, 80)
- [ ] SSL certificate installed (use Let's Encrypt)
- [ ] Domain configured and DNS propagated
- [ ] Reverse proxy (nginx/Apache) configured
- [ ] Process manager installed (systemd, supervisor, PM2)

## Application Deployment

### Heroku
- [ ] Heroku app created
- [ ] Git remote configured: `git remote add heroku <url>`
- [ ] Procfile verified
- [ ] runtime.txt verified (Python version)
- [ ] Config vars set: `heroku config:set FLASK_ENV=production SECRET_KEY=<key>`
- [ ] Deploy: `git push heroku main`

### Docker
- [ ] Dockerfile created and tested
- [ ] docker-compose.yml configured
- [ ] Built and tested locally: `docker build -t ml-dashboard .`
- [ ] Verified image runs: `docker run -p 5000:5000 ml-dashboard`
- [ ] Docker registry/repository configured (Docker Hub, ECR, GCR)
- [ ] Pushed to registry

### Traditional Server
- [ ] Application directory created: `/opt/ml-dashboard/`
- [ ] Venv activated and dependencies installed
- [ ] Gunicorn installed
- [ ] Systemd service file created
- [ ] Service enabled: `systemctl enable ml-dashboard`
- [ ] Service started: `systemctl start ml-dashboard`
- [ ] Nginx/Apache proxy configured and reloaded

## Monitoring & Logging

- [ ] Application logs configured and monitored
- [ ] Error tracking setup (Sentry, Rollbar, etc.)
- [ ] Uptime monitoring configured (UptimeRobot, Pingdom)
- [ ] Alerting configured for errors and downtime
- [ ] CPU, memory, disk monitoring enabled
- [ ] Database query monitoring (if applicable)
- [ ] Slow query logging enabled

## Backup & Recovery

- [ ] Uploads directory backup plan
- [ ] Trained models backup plan
- [ ] Database backup scheduled (if applicable)
- [ ] Backup retention policy defined
- [ ] Disaster recovery plan documented
- [ ] Backup restoration tested

## Performance

- [ ] Response time benchmarked
- [ ] Load testing performed
- [ ] Static files served by reverse proxy/CDN
- [ ] Database indexes created
- [ ] Query optimization complete
- [ ] Caching strategy implemented
- [ ] Gzip compression enabled

## Post-Deployment

- [ ] Access application via domain
- [ ] Test user registration/login (if implemented)
- [ ] Test file upload
- [ ] Test model training
- [ ] Monitor logs for errors
- [ ] Performance metrics verified
- [ ] SSL certificate working correctly
- [ ] All features tested in production

## Documentation

- [ ] Deployment procedure documented
- [ ] Environment variables documented
- [ ] Runbook created for common issues
- [ ] Team trained on deployment process
- [ ] On-call procedure established

## Maintenance

- [ ] Update schedule established
- [ ] Dependency update process documented
- [ ] Security patch procedure documented
- [ ] Database maintenance scheduled
- [ ] Log rotation configured
- [ ] Cleanup routines scheduled

---

**Deployment Date**: _______________
**Deployed By**: _______________
**Version**: _______________
**Notes**: _______________
