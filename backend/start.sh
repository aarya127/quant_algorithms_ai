#!/bin/bash

# Stock Predictor UI Launch Script

echo "🚀 Starting Stock Predictor Web Application..."
echo ""

# Activate virtual environment
if [ -f "../.venv/bin/activate" ]; then
    source ../.venv/bin/activate
    echo "✓ Virtual environment activated"
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "❌ Virtual environment not found!"
    echo "Please run: python -m venv .venv"
    exit 1
fi

# Check if Flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo "❌ Flask not found! Installing dependencies..."
    pip install -r requirements.txt
fi

# Check API keys
if [ ! -f "../keys.txt" ]; then
    echo "⚠️  Warning: keys.txt not found in parent directory!"
    echo "Make sure to add your API keys before using the app."
fi

echo ""
echo "🌐 Starting Flask server on http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""

# Start the Flask app
export FLASK_APP=app.py
export FLASK_ENV=development
python app.py
