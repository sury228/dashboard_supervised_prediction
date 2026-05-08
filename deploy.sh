#!/bin/bash
# Production deployment script

set -e

echo "🚀 ML Dashboard Production Deployment"
echo "===================================="

# Check required tools
command -v python &> /dev/null || { echo "❌ Python not found"; exit 1; }
command -v pip &> /dev/null || { echo "❌ pip not found"; exit 1; }

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "✅ Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found! Creating from .env.example..."
    cp .env.example .env
    echo "⚠️  Please edit .env with your configuration!"
    exit 1
fi

# Create required directories
echo "📁 Creating required directories..."
mkdir -p logs uploads models

# Generate SECRET_KEY if not set
if grep -q "SECRET_KEY=your-secure-random-key-here" .env; then
    echo "🔐 Generating SECRET_KEY..."
    SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')
    sed -i "s/SECRET_KEY=.*/SECRET_KEY=$SECRET_KEY/" .env
fi

# Set production environment
export FLASK_ENV=production

echo ""
echo "✅ Deployment setup complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Verify settings in .env"
echo "   2. Run with Gunicorn:"
echo "      gunicorn wsgi:app --workers=4 --bind=0.0.0.0:5000"
echo ""
echo "   Or run with Docker:"
echo "      docker-compose up -d"
echo ""
