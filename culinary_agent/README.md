# Culinary Agent - Smart Daily Meal Planner

> Production-ready generative AI system for intelligent meal planning with agentic RAG, FastAPI backend, and Streamlit frontend.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

The Culinary Agent is a sophisticated meal planning system that combines:
- **Vector Search (FAISS)** for semantic recipe retrieval
- **LLM-powered adaptation** to rewrite recipes for dietary constraints
- **Strict validation** to ensure compliance with dietary requirements
- **Web search fallback** via DuckDuckGo for missing recipes
- **Smart meal planning** with targeted updates for breakfast, lunch, and dinner
- **REST API** for programmatic access via FastAPI
- **Web Interface** for user-friendly interactions via Streamlit

Built with clean architecture principles, proper type hinting, and comprehensive error handling.

## Features

✅ **Daily Meal Planning**: Generate complete meal plans (Breakfast, Lunch, Dinner)  
✅ **Selective Meal Updates**: Modify individual meals while preserving others  
✅ **Semantic Recipe Search**: FAISS-indexed vector search using BAAI/bge-m3 embeddings  
✅ **Dietary Adaptation**: LLM-powered recipe modification (vegan, gluten-free, keto, etc.)  
✅ **Compliance Validation**: Deterministic PASS/FAIL checks for ingredient safety  
✅ **Fallback Search**: DuckDuckGo web search when local DB has no matches  
✅ **Stateful Sessions**: Persistent meal plan memory across sessions  
✅ **Robust Error Handling**: Automatic retry with exponential backoff for transient failures  
✅ **REST API**: FastAPI backend with automatic OpenAPI documentation  
✅ **Web Interface**: Streamlit frontend for interactive meal planning  
✅ **Production Ready**: CORS support, health checks, and proper error responses  

## Project Structure

```
culinary_agent/
├── api.py                         # FastAPI REST API server
├── frontend/
│   └── app.py                     # Streamlit web interface
├── data/
│   └── recipes/                   # Recipe data files
│       ├── recipes_for_embeddings.jsonl
│       └── full_format_recipes.json
├── indices/
│   └── recipe_index.faiss         # FAISS vector index
├── src/
│   ├── __init__.py
│   ├── main.py                    # CLI entry point
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── core.py                # CulinaryAgent & factory
│   │   └── prompts.py             # SYSTEM_PROMPT definitions
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── retriever.py           # RecipeRetrieverTool (FAISS)
│   │   ├── adapter.py             # RecipeAdapterTool (LLM)
│   │   ├── validator.py           # RecipeValidatorTool (LLM)
│   │   └── web_search.py          # WebSearchTool (DuckDuckGo)
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── decorators.py          # @robust_llm_call decorator
│   │   └── state_manager.py       # StateManager for sessions
│   ├── indexing/
│   │   ├── __init__.py
│   │   └── build_index.py         # RecipeIndexBuilder
│   └── config/
│       ├── __init__.py
│       └── settings.py            # Constants & paths
├── data/                          # Recipe data files
│   └── .gitkeep
├── indices/                       # FAISS indices
│   └── .gitkeep
├── tests/
│   ├── __init__.py
│   ├── test_tools.py
│   └── test_agent.py
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variables template
├── README.md                      # This file
└── REFACTORING_PLAN.md           # Project improvement roadmap
```

**Note**: The project now follows a clean architecture with organized data, frontend, and backend components. For Docker deployment and advanced features, see [REFACTORING_PLAN.md](REFACTORING_PLAN.md).

## Installation

### Prerequisites

- Python 3.8+
- `pip` or `conda`
- Recipe data files (`recipes_for_embeddings.jsonl`, `full_format_recipes.json`)
- Pre-built FAISS index (`recipe_index.faiss`)

### Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd /path/to/meal_planner/culinary_agent
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys (HuggingFace, Langfuse, etc.)
   ```

4. **Ensure data files are in the project root:**
   - `recipes_for_embeddings.jsonl`
   - `full_format_recipes.json`
   - `recipe_index.faiss`

   *If missing, run the index builder (see below).*
   
   **Note**: Data files are now organized in `data/recipes/` and `indices/` directories within the project.

## Usage

### Three Ways to Use the System

1. **CLI Mode** - Command-line interface for developers
2. **API Mode** - REST API for programmatic access
3. **Web Interface** - Streamlit frontend for end users

---

### 1. CLI Mode

**Quick Start:**
```bash
cd /path/to/meal_planner/culinary_agent
./scripts/run_cli.sh
```

**Alternative Methods:**

**Option A** - Using the wrapper script (deprecated but works):
```bash
cd /path/to/meal_planner/culinary_agent/src
python main.py
```
*Note: This shows a deprecation warning but still redirects to the new CLI*

**Option B** - Direct module execution (recommended):
```bash
cd /path/to/meal_planner/culinary_agent
python -m src.inference.cli
```

**Option C** - From Python REPL:
```python
from src.inference.cli import main
main()
```

**Example Session:**

```
============================================================
 🍳 CULINARY AGENT - Smart Daily Meal Planner
============================================================

Initializing agent...
✓ Culinary Agent ready!

------------------------------------------------------------
 📋 PLANNING MODE SELECTION:
 1. Generate Full Day Plan (Breakfast, Lunch, Dinner)
 2. Modify/Generate ONLY Breakfast
 3. Modify/Generate ONLY Lunch
 4. Modify/Generate ONLY Dinner
 c. Clear Session (Start Fresh)
 q. Quit
------------------------------------------------------------
Select an option (1-4, c, q): 1

Enter dietary preferences (e.g., 'vegan', 'high protein'): vegan high protein

============================================================
🤔 Agent is thinking... [Task: Create a full daily meal plan. Constraint: vegan high protein]
============================================================

🍽️  AGENT RESPONSE:
============================================================

[Complete meal plan with Breakfast, Lunch, and Dinner appears here...]

------------------------------------------------------------
 📋 PLANNING MODE SELECTION:
 1. Generate Full Day Plan (Breakfast, Lunch, Dinner)
 2. Modify/Generate ONLY Breakfast
 3. Modify/Generate ONLY Lunch
 4. Modify/Generate ONLY Dinner
 c. Clear Session (Start Fresh)
 q. Quit
------------------------------------------------------------
Select an option (1-4, c, q): 3

Enter dietary preferences (e.g., 'vegan', 'high protein'): gluten-free

[Only lunch gets updated, breakfast and dinner remain unchanged...]
```

**Menu Options:**

1. **Generate Full Day Plan** - Creates a complete meal plan with breakfast, lunch, and dinner
2. **Modify ONLY Breakfast** - Updates breakfast while keeping lunch and dinner from previous plan
3. **Modify ONLY Lunch** - Updates lunch while keeping breakfast and dinner from previous plan
4. **Modify ONLY Dinner** - Updates dinner while keeping breakfast and lunch from previous plan
5. **Clear Session** - Removes saved state and starts fresh
6. **Quit** - Exits the application (saves current state)

---

### 2. API Mode (FastAPI Backend)

Start the REST API server:

```bash
cd /path/to/meal_planner/culinary_agent
uvicorn src.api.server:app --reload
```

Or run directly:
```bash
python -m src.api.server
```

**Access the API:**
- **Interactive Docs**: http://127.0.0.1:8000/docs (Swagger UI)
- **Alternative Docs**: http://127.0.0.1:8000/redoc
- **Root**: http://127.0.0.1:8000/
- **Health Check**: http://127.0.0.1:8000/health

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information and available endpoints |
| `/health` | GET | Health check - returns agent status |
| `/generate-plan` | POST | Generate or modify meal plans |

**Example API Usage:**

**1. Health Check** - Verify the API is running and agent is ready:
```bash
curl http://127.0.0.1:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "version": "1.0.0"
}
```

**2. Generate Full Day Plan** - Create a complete meal plan from scratch:
```bash
curl -X POST http://127.0.0.1:8000/generate-plan \
  -H "Content-Type: application/json" \
  -d '{
    "diet_constraints": "vegan, high protein",
    "mode": "Full Day"
  }'
```

Expected response:
```json
{
  "status": "success",
  "result": "==================== BREAKFAST ====================\nTofu Scramble with Spinach\n...\n==================== LUNCH ====================\nQuinoa Buddha Bowl\n...\n==================== DINNER ====================\nLentil Curry with Brown Rice\n...",
  "mode": "Full Day",
  "diet_constraints": "vegan, high protein"
}
```

**3. Update Only Breakfast** - Modify just breakfast while preserving lunch and dinner:
```bash
# First, save your current plan to a variable or file
CURRENT_PLAN=$(cat << 'EOF'
==================== BREAKFAST ====================
Tofu Scramble with Spinach
...
==================== LUNCH ====================
Quinoa Buddha Bowl
...
==================== DINNER ====================
Lentil Curry with Brown Rice
...
EOF
)

# Then update only breakfast
curl -X POST http://127.0.0.1:8000/generate-plan \
  -H "Content-Type: application/json" \
  -d "{
    \"diet_constraints\": \"gluten-free, high protein\",
    \"mode\": \"Breakfast\",
    \"previous_plan\": \"$CURRENT_PLAN\"
  }"
```

**4. Update Specific Meals** - Examples for lunch and dinner:
```bash
# Update only lunch
curl -X POST http://127.0.0.1:8000/generate-plan \
  -H "Content-Type: application/json" \
  -d '{
    "diet_constraints": "keto, dairy-free",
    "mode": "Lunch",
    "previous_plan": "..."
  }'

# Update only dinner
curl -X POST http://127.0.0.1:8000/generate-plan \
  -H "Content-Type: application/json" \
  -d '{
    "diet_constraints": "paleo, nut-free",
    "mode": "Dinner",
    "previous_plan": "..."
  }'
```

**Python Client Examples:**

**Basic Usage:**
```python
import requests

# 1. Check API health
health = requests.get("http://127.0.0.1:8000/health")
print(f"API Status: {health.json()['status']}")

# 2. Generate full day meal plan
response = requests.post(
    "http://127.0.0.1:8000/generate-plan",
    json={
        "diet_constraints": "keto, dairy-free",
        "mode": "Full Day"
    }
)

if response.status_code == 200:
    data = response.json()
    print(f"Status: {data['status']}")
    print(f"Mode: {data['mode']}")
    print(f"Constraints: {data['diet_constraints']}")
    print(f"\nMeal Plan:\n{data['result']}")
else:
    print(f"Error: {response.json()['detail']}")
```

**Advanced Usage - Iterative Meal Planning:**
```python
import requests

class MealPlannerClient:
    """Client for interacting with the Culinary Agent API."""
    
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url
        self.current_plan = None
    
    def generate_full_plan(self, diet_constraints):
        """Generate a complete meal plan."""
        response = requests.post(
            f"{self.base_url}/generate-plan",
            json={
                "diet_constraints": diet_constraints,
                "mode": "Full Day"
            }
        )
        if response.status_code == 200:
            dasrc/frontend/app.py
```

**Access the Interface:**
- Open http://localhost:8501 in your browser

**Features:**
- 🎮 **Interactive Controls**: Easy-to-use sidebar for diet preferences
- 📋 **Mode Selection**: Choose full day or specific meal updates
- 💾 **Session Memory**: Automatically preserves your current plan
- ✏️ **Manual Editing**: Override or paste previous plans
- 🚀 **One-Click Generation**: Execute agent with a single button

**Prerequisites:**
- FastAPI backend must be running at http://127.0.0.1:8000
- Install frontend dependencies: `pip install streamlit requests`

**Detailed Workflow:**

**First Time Setup:**
1. **Start the backend** (in Terminal 1):
   ```bash
   cd /path/to/meal_planner/culinary_agent
   uvicorn src.api.server:app --reload
   ```
   Wait for: `✅ Agent is ready to cook!`

2. **Start the frontend** (in Terminal 2):
   ```bash
   streamlit run src/frontend/app.py
   ```
   Browser should auto-open to http://localhost:8501

**Using the Interface:**

**Scenario 1: Generate Your First Meal Plan**
1. In the sidebar, enter dietary constraints:
   ```
   vegan, high protein, gluten-free
   ```
2. Ensure "Full Day Plan" is selected
3. Click "🚀 Execute Agent"
4. Wait 10-30 seconds for the agent to generate your plan
5. View the complete plan with Breakfast, Lunch, and Dinner

**Scenario 2: Update a Specific Meal**
1. After generating a plan, new options appear:
   - "Full Day Plan (Overwrite)"
   - "Update Breakfast Only"
   - "Update Lunch Only"
   - "Update Dinner Only"
2. Select "Update Breakfast Only"
3. Change constraints to (e.g.):
   ```
   vegan, high protein, quick recipes
   ```
4. Click "🚀 Execute Agent"
5. Only breakfast updates; lunch and dinner remain unchanged

**Scenario 3: Iterative Refinement**
1. Start with: `vegetarian, budget-friendly`
2. Review the plan
3. Update breakfast: `vegetarian, budget-friendly, quick prep`
4. Review breakfast
5. Update lunch: `vegetarian, budget-friendly, meal prep friendly`
6. Review lunch
7. Keep refining until satisfied

**Scenario 4: Manual Override**
1. Generate any meal plan
2. Expand "📝 Manually Edit / Paste Old Plan"
3. Edit the text directly (e.g., add your own recipe)
4. Click "Save Manual Edits"
5. Continue using the agent with your custom plan

**Tips for Best Results:**
- Be specific with constraints: `"low-carb, dairy-free, under 30 minutes"`
- Use commas to separate multiple requirements
- Start with a full day plan before making updates
- The agent remembers your plan within the session
- Use manual edit to save custom recipes between updates
            raise Exception(f"API Error: {response.json()}")

# Example workflow
client = MealPlannerClient()

# Step 1: Generate initial plan
print("Generating initial meal plan...")
plan1 = client.generate_full_plan("vegan, high protein")
print(f"✓ Generated full day plan\n")

# Step 2: Update breakfast to be gluten-free
print("Updating breakfast to gluten-free...")
plan2 = client.update_meal("Breakfast", "vegan, high protein, gluten-free")
print(f"✓ Updated breakfast (lunch and dinner preserved)\n")

# Step 3: Make dinner low-carb
print("Making dinner low-carb...")
plan3 = client.update_meal("Dinner", "vegan, high protein, low-carb")
print(f"✓ Updated dinner (breakfast and lunch preserved)\n")

# Display final plan
print("="*60)
print("FINAL MEAL PLAN")
print("="*60)
print(client.current_plan)
```

**Handling Errors and Retries:**
```python
import requests
import time
from typing import Optional

def generate_plan_with_retry(
    diet_constraints: str,
    mode: str = "Full Day",
    previous_plan: Optional[str] = None,
    max_retries: int = 3
):
    """Generate meal plan with automatic retry on failure."""
    
    url = "http://127.0.0.1:8000/generate-plan"
    payload = {
        "diet_constraints": diet_constraints,
        "mode": mode
    }
    if previous_plan:
        payload["previous_plan"] = previous_plan
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=60)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 503:
                # Agent not ready, wait and retry
                print(f"Agent initializing... (attempt {attempt + 1}/{max_retries})")
                time.sleep(10)
            else:
                # Other error, raise immediately
                response.raise_for_status()
        
        except requests.exceptions.Timeout:
            print(f"Request timed out (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(5)
        
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
    
    raise Exception("Max retries exceeded")

# Usage
result = generate_plan_with_retry(
    diet_constraints="mediterranean, heart-healthy",
    mode="Full Day"
)
print(result['result'])
```

**Batch Processing Multiple Dietary Preferences:**
```python
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

def generate_plan_for_diet(diet_constraints):
    """Generate a meal plan for specific dietary constraints."""
    response = requests.post(
        "http://127.0.0.1:8000/generate-plan",
        json={
            "diet_constraints": diet_constraints,
            "mode": "Full Day"
        }
    )
    if response.status_code == 200:
        return diet_constraints, response.json()['result']
    else:
        return diet_constraints, f"Error: {response.status_code}"

# Generate plans for multiple diets concurrently
diets = [
    "vegan, high protein",
    "keto, dairy-free",
    "paleo, nut-free",
    "mediterranean, low-sodium",
    "gluten-free, vegetarian"
]

plans = {}
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = {executor.submit(generate_plan_for_diet, diet): diet 
               for diet in diets}
    
    for future in as_completed(futures):
        diet_constraints, plan = future.result()
        plans[diet_constraints] = plan
        print(f"✓ Generated plan for: {diet_constraints}")

# Save all plans
import json
with open("meal_plans.json", "w") as f:
    json.dump(plans, f, indent=2)
```
```

**Request Schema:**
```json
{
  "diet_constraints": "string (required)",
  "mode": "Full Day | Breakfast | Lunch | Dinner",
  "previous_plan": "string (optional - for updates)"
}
```

**Response Schema:**
```json
{
  "status": "success | error",
  "result": "string - the meal plan text",
  "mode": "string - what was generated",
  "diet_constraints": "string - constraints used"
}
```

---

### 3. Web Interface (Streamlit Frontend)

Start the Streamlit web interface:

```bash
cd /path/to/meal_planner/culinary_agent
streamlit run frontend/app.py
```

**Access the Interface:**
- Open http://localhost:8501 in your browser

**Features:**
- 🎮 **Interactive Controls**: Easy-to-use sidebar for diet preferences
- 📋 **Mode Selection**: Choose full day or specific meal updates
- 💾 **Session Memory**: Automatically preserves your current plan
- ✏️ **Manual Editing**: Override or paste previous plans
- 🚀 **One-Click Generation**: Execute agent with a single button

**Prerequisites:**
- FastAPI backend must be running at http://127.0.0.1:8000
- Install frontend dependencies: `pip install streamlit requests`

**Typical Workflow:**
1. Start backend: `uvicorn src.api.server:app --reload`
2. Start frontend: `streamlit run src/frontend/app.py`
3. Enter dietary constraints (e.g., "vegan, high protein")
4. Click "Execute Agent"
5. View generated plan
6. Make selective updates as needed

---

### Menu Options

1. **Generate Full Day Plan** - Creates a complete meal plan with breakfast, lunch, and dinner
2. **Modify ONLY Breakfast** - Updates breakfast while keeping lunch and dinner from previous plan
3. **Modify ONLY Lunch** - Updates lunch while keeping breakfast and dinner from previous plan
4. **Modify ONLY Dinner** - Updates dinner while keeping breakfast and lunch from previous plan
5. **Clear Session** - Removes saved state and starts fresh
6. **Quit** - Exits the application (saves current state)

### Building the FAISS Index

If `recipe_index.faiss` doesn't exist:
```bash
cd /path/to/meal_planner/culinary_agent
python -m src.indexing.build_index
```

This will:
1. Load recipes from `recipes_for_embeddings.jsonl`
2. Generate embeddings using `BAAI/bge-m3`
3. Build and save a FAISS index with cosine similarity

## Configuration

Edit `src/config/settings.py` to customize:

```python
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"  # LLM model
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"     # Embedding model
RECIPE_EMBEDDINGS_FILE = Path("recipes_for_embeddings.jsonl")
FULL_RECIPES_FILE = Path("full_format_recipes.json")
FAISS_INDEX_FILE = Path("recipe_index.faiss")
```

## Architecture

### Components

**RecipeRetrieverTool** ([src/tools/retriever.py](src/tools/retriever.py))  
- Uses FAISS for vector search over recipe embeddings
- Returns formatted recipe with title, ingredients, and directions
- Graceful fallback on errors

**RecipeAdapterTool** ([src/tools/adapter.py](src/tools/adapter.py))  
- LLM-powered recipe rewriting for dietary constraints
- Preserves dish identity while substituting forbidden ingredients
- Handles response formatting robustly

**RecipeValidatorTool** ([src/tools/validator.py](src/tools/validator.py))  
- Deterministic PASS/FAIL validation
- Truncates long recipes to avoid context limits
- Defaults to FAIL on errors for safety

**WebSearchTool** ([src/tools/web_search.py](src/tools/web_search.py))  
- DuckDuckGo integration for web recipe search
- Returns top 3 results formatted for LLM consumption

**CulinaryAgent** ([src/agent/core.py](src/agent/core.py))  
- Orchestrates all tools using `smolagents.CodeAgent`
- Retry logic for transient server errors
- Stateful conversation management

### Error Handling

The `@robust_llm_call` decorator ([src/utils/decorators.py](src/utils/decorators.py)) provides:
- Exponential backoff retry (up to 5 attempts)
- Detection of transient errors (503, 404, connection errors)
- Immediate failure for code errors

## Development

### Running Tests

```bash
cd /path/to/meal_planner/culinary_agent
pytest tests/
```

Or with verbose output:
```bash
pytest tests/ -v
```

### Running All Components Together

**Option 1: Manual Startup**
```bash
# Terminal 1: Start backend
cd culinary_agent
uvicorn src.api.server:app --reload

# Terminal 2: Start frontend (in same directory)
streamlit run src/frontend/app.py

# Terminal 3: Test API
curl http://127.0.0.1:8000/health
```

**Option 2: Using Background Processes**
```bash
# Start backend in background
cd culinary_agent
uvicorn src.api.server:app --reload &

# Wait for startup
sleep 5

# Start frontend in same directory
streamlit run src/frontend/app.py
```

### Development Workflow

1. **Make code changes** in `src/` directory
2. **Run tests** to verify: `pytest tests/ -v`
3. **Test API** endpoint changes: `curl http://localhost:8000/docs`
4. **Test frontend** integration: Open http://localhost:8501

### Debugging

**Enable Debug Logging:**
```python
# In api.py or main.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Check Agent Initialization:**
```bash
curl http://127.0.0.1:8000/health
# Should return: {"status": "healthy", "model": "Qwen/Qwen2.5-14B-Instruct", ...}
```

**Test Individual Tools:**
```python
# In Python REPL
from src.tools.retriever import RecipeRetrieverTool
tool = RecipeRetrieverTool()
result = tool.forward(query="vegan pasta")
print(result)
```

### Code Quality

**Run linters:**
```bash
# Install dev dependencies
pip install black flake8 mypy

# Format code
black src/ tests/

# Check style
flake8 src/ tests/

# Type checking
mypy src/
```

### Adding New Tools

1. Create a new file in `src/tools/`
2. Inherit from `smolagents.Tool`
3. Define `name`, `description`, `inputs`, and `output_type`
4. Implement `forward(self, **kwargs)` method
5. Register in `agent/core.py` `_initialize_tools()`

**Example:**
```python
# src/tools/nutrition_calculator.py
from smolagents import Tool

class NutritionCalculatorTool(Tool):
    name = "calculate_nutrition"
    description = "Calculates nutritional information for a recipe"
    inputs = {"recipe_text": {"type": "string", "description": "Full recipe text"}}
    output_type = "string"
    
    def forward(self, recipe_text: str) -> str:
        # Your implementation
        return f"Calories: {calories}, Protein: {protein}g..."

# Then in src/agent/core.py
def _initialize_tools(self):
    from tools.nutrition_calculator import NutritionCalculatorTool
    return [
        RecipeRetrieverTool(...),
        # ... existing tools
        NutritionCalculatorTool()  # Add here
    ]
```

## Deployment

### Production Checklist

- [ ] Set environment variables in `.env`
- [ ] Move data files to `data/` directory (see [REFACTORING_PLAN.md](REFACTORING_PLAN.md))
- [ ] Build FAISS index with production data
- [ ] Run test suite: `pytest tests/`
- [ ] Configure proper CORS origins in `api.py`
- [ ] Set up reverse proxy (nginx) for SSL
- [ ] Configure logging and monitoring
- [ ] Set up automatic restarts (systemd/supervisor)

### Quick Deploy (Development)

```bash
# 1. Clone repository
git clone <repository-url>
cd meal_planner/culinary_agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment
cp .env.example .env
# Edit .env with your API keys

# 4. Data files should already be in data/ and indices/ directories

# 5. Start services
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 &
streamlit run src/frontend/app.py --server.port 8501
```

### Docker Deployment (Coming Soon)

For containerized deployment with Docker, see the [REFACTORING_PLAN.md](REFACTORING_PLAN.md) for:
- Dockerfile configuration
- docker-compose setup
- Multi-service orchestration
- Volume management for data persistence

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'agent'`  
**Solution**: Ensure you're running from the correct directory or using absolute imports. The `api.py` adds `src/` to the Python path.

**Issue**: `FileNotFoundError: recipe_index.faiss`  
**Solution**: Build the FAISS index first:
```bash
python -m culinary_agent_backend.src.indexing.build_index
```

**Issue**: API returns 503 "Agent not ready"  
**Solution**: Check logs for initialization errors. The agent loads models on startup which can take 30-60 seconds.

**Issue**: Frontend shows "Connection Failed"  
**Solution**: Ensure backend is running at http://127.0.0.1:8000. Check `API_URL` in `front_end.py`.

**Issue**: Model download fails  
**Solution**: Set HuggingFace token:
```bash
export HF_TOKEN=your_token_here
# Or add to .env file
```

**Issue**: Out of memory when loading models  
**Solution**: Requires ~16GB RAM for Qwen2.5-14B. Consider:
- Using a smaller model (edit `settings.py`)
- Running on a machine with more RAM
- Using model quantization

## Performance Optimization

### Model Loading
- **First Request**: ~30-60 seconds (model loading)
- **Subsequent Requests**: ~5-15 seconds (inference only)
- **Caching**: Models cached in memory after first load

### FAISS Search
- **Index Size**: ~50MB for 10k recipes
- **Search Time**: <100ms for semantic search
- **Memory**: Loaded once at startup

### API Performance
- **Concurrent Requests**: Supported via async handlers
- **Rate Limiting**: Not implemented (add in production)
- **Caching**: Not implemented (consider Redis for production)

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | Qwen/Qwen2.5-14B-Instruct | Recipe generation & adaptation |
| **Embeddings** | BAAI/bge-m3 | Semantic recipe search |
| **Vector DB** | FAISS | Fast similarity search |
| **Agent Framework** | smolagents | Tool orchestration |
| **Backend API** | FastAPI | REST API server |
| **Frontend** | Streamlit | Web interface |
| **Web Search** | DuckDuckGo (ddgs) | Fallback recipe search |
| **Testing** | pytest | Unit & integration tests |

## Architecture Diagrams

### System Flow
```
User Input (CLI/API/Web)
        ↓
CulinaryAgent (smolagents)
        ↓
    ┌───┴───┐
    ↓       ↓       ↓       ↓
Retriever Adapter Validator WebSearch
(FAISS)   (LLM)   (LLM)    (DuckDuckGo)
    ↓       ↓       ↓       ↓
    └───┬───┘
        ↓
Meal Plan Output
```

### API Architecture
```
Frontend (Streamlit)
        ↓ HTTP
FastAPI Server (api.py)
        ↓
CulinaryAgent
        ↓
[Tools] → FAISS Index
        → LLM Model
        → Web Search
```

## Original Notebooks

The production code was refactored from:
- [`build_index.ipynb`](../build_index.ipynb) → `src/indexing/build_index.py`
- [`agentic_rag.ipynb`](../agentic_rag.ipynb) → `src/agent/`, `src/tools/`, `src/utils/`

**Non-destructive refactoring**: Original notebooks remain untouched for reference.

## Roadmap

### Completed ✅
- [x] Core agent implementation
- [x] FAISS-based recipe retrieval
- [x] LLM-powered recipe adaptation
- [x] Dietary validation
- [x] CLI interface with menu system
- [x] FastAPI REST API
- [x] Streamlit web interface
- [x] Session state management
- [x] Comprehensive test suite

### Planned 🚧
- [ ] Docker containerization
- [ ] Frontend integration into project structure
- [ ] Data files organization (move to `data/` directory)
- [ ] Nutrition information display
- [ ] Meal plan export (PDF, JSON)
- [ ] User authentication
- [ ] Meal plan history
- [ ] Favorite recipes
- [ ] Shopping list generation
- [ ] API rate limiting
- [ ] Prometheus metrics
- [ ] CI/CD pipeline

See [REFACTORING_PLAN.md](REFACTORING_PLAN.md) for detailed implementation plan.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes and add tests
4. Run test suite: `pytest tests/`
5. Run linters: `black . && flake8 .`
6. Commit changes: `git commit -am 'Add feature'`
7. Push to branch: `git push origin feature/your-feature`
8. Submit a pull request

## Support

- **Issues**: Report bugs via GitHub Issues
- **Documentation**: See this README and [REFACTORING_PLAN.md](REFACTORING_PLAN.md)
- **API Docs**: http://127.0.0.1:8000/docs (when server is running)

## Acknowledgments

- **smolagents**: Agent framework by HuggingFace
- **FAISS**: Vector similarity search by Meta AI
- **Qwen**: Language model by Alibaba Cloud
- **Recipe Dataset**: [Source to be added]

## License

This project is licensed under the MIT License.