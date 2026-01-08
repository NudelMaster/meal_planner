# Culinary Agent - Smart Daily Meal Planner

Production-ready modular Python system for an intelligent daily meal planning agent with FastAPI backend and Streamlit frontend.

## Overview

The Culinary Agent Backend is a sophisticated meal planning system that combines:
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

From the `culinary_agent/src` directory:

```bash
cd /path/to/meal_planner/culinary_agent/src
python main.py
```

Or as a module from the project root:

```bash
cd /path/to/meal_planner
python -m culinary_agent.src.main
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
uvicorn api:app --reload
```

Or run directly:
```bash
python api.py
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

```bash
# Health check
curl http://127.0.0.1:8000/health

# Generate full day plan
curl -X POST http://127.0.0.1:8000/generate-plan \
  -H "Content-Type: application/json" \
  -d '{
    "diet_constraints": "vegan, high protein",
    "mode": "Full Day"
  }'

# Update only breakfast (requires previous_plan)
curl -X POST http://127.0.0.1:8000/generate-plan \
  -H "Content-Type: application/json" \
  -d '{
    "diet_constraints": "gluten-free",
    "mode": "Breakfast",
    "previous_plan": "BREAKFAST:\n..."
  }'
```

**Python Client Example:**

```python
import requests

# Generate meal plan
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
    print(f"Plan:\n{data['result']}")
else:
    print(f"Error: {response.json()['detail']}")
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
1. Start backend: `uvicorn api:app --reload`
2. Start frontend: `streamlit run front_end.py`
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
uvicorn api:app --reload

# Terminal 2: Start frontend (in same directory)
streamlit run frontend/app.py

# Terminal 3: Test API
curl http://127.0.0.1:8000/health
```

**Option 2: Using Background Processes**
```bash
# Start backend in background
cd culinary_agent
uvicorn api:app --reload &

# Wait for startup
sleep 5

# Start frontend in same directory
streamlit run frontend/app.py
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
uvicorn api:app --host 0.0.0.0 --port 8000 &
streamlit run frontend/app.py --server.port 8501
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