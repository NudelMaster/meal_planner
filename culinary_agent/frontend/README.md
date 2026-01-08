# Culinary Agent Frontend

Streamlit-based web interface for the Culinary Agent meal planning system.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the frontend (ensure backend is running first)
streamlit run app.py
```

## Configuration

The frontend connects to the FastAPI backend at:
- **Default**: `http://127.0.0.1:8000`
- **Configurable**: Edit `API_URL` in `app.py`

## Features

- 🎮 Interactive sidebar controls for dietary preferences
- 📋 Mode selection (Full Day, Breakfast, Lunch, Dinner)
- 💾 Automatic session state management
- ✏️ Manual plan editing capability
- 🚀 One-click meal plan generation

## Prerequisites

- Backend API running at `http://127.0.0.1:8000`
- Python 3.8+
- Streamlit 1.28.0+

## Usage

1. Start the backend: `cd .. && uvicorn api:app --reload`
2. Start the frontend: `streamlit run app.py`
3. Open http://localhost:8501 in your browser
4. Enter dietary constraints and generate meal plans

## Development

To modify the frontend:
- Edit `app.py` for core logic
- Add components in a `components/` directory for modularity
- Update `requirements.txt` for new dependencies
