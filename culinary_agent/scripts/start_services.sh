#!/bin/bash
# Start API and Streamlit services

set -e

echo "🚀 Starting Culinary Agent Services"
echo "===================================="

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check if API is already running
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  API server already running on port 8000"
else
    echo "Starting API server on port 8000..."
    uvicorn src.api.server:app --host 0.0.0.0 --port 8000 &
    API_PID=$!
    echo "✓ API server started (PID: $API_PID)"
fi

# Wait a moment for API to start
sleep 2

# Check if Streamlit is already running
if lsof -Pi :8501 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Streamlit already running on port 8501"
else
    echo "Starting Streamlit frontend on port 8501..."
    streamlit run src/frontend/app.py --server.port 8501 &
    STREAMLIT_PID=$!
    echo "✓ Streamlit started (PID: $STREAMLIT_PID)"
fi

echo ""
echo "✅ Services started!"
echo ""
echo "📍 Access points:"
echo "  API:        http://localhost:8000"
echo "  API Docs:   http://localhost:8000/docs"
echo "  Frontend:   http://localhost:8501"
echo ""
echo "To stop services, run:"
echo "  pkill -f uvicorn"
echo "  pkill -f streamlit"
