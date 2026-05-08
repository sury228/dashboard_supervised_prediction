@echo off
REM Production deployment script for Windows

echo.
echo ML Dashboard Production Deployment
echo ===================================
echo.

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found
    exit /b 1
)

REM Create virtual environment if not exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Check for .env file
if not exist ".env" (
    echo .env file not found! Creating from .env.example...
    copy .env.example .env
    echo Please edit .env with your configuration!
    exit /b 1
)

REM Create required directories
echo Creating required directories...
if not exist logs mkdir logs
if not exist uploads mkdir uploads
if not exist models mkdir models

echo.
echo Deployment setup complete!
echo.
echo Next steps:
echo   1. Verify settings in .env
echo   2. Run with Gunicorn:
echo      gunicorn wsgi:app --workers=4 --bind=0.0.0.0:5000
echo.
echo   Or run with Docker:
echo      docker-compose up -d
echo.
