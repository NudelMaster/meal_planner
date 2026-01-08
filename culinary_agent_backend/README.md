# Culinary Agent Backend - Smart Daily Meal Planner

Production-ready modular Python backend for an intelligent daily meal planning agent, refactored from experimental Jupyter notebooks.

## Overview

The Culinary Agent Backend is a sophisticated meal planning system that combines:
- **Vector Search (FAISS)** for semantic recipe retrieval
- **LLM-powered adaptation** to rewrite recipes for dietary constraints
- **Strict validation** to ensure compliance with dietary requirements
- **Web search fallback** via DuckDuckGo for missing recipes
- **Smart meal planning** with targeted updates for breakfast, lunch, and dinner

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

## Project Structure

```
culinary_agent_backend/
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
├── data/                          # Placeholder for data files
├── indices/                       # Placeholder for FAISS indices
├── tests/
│   ├── __init__.py
│   ├── test_tools.py
│   └── test_agent.py
├── requirements.txt
├── .env.example
└── README.md
```

## Installation

### Prerequisites

- Python 3.8+
- `pip` or `conda`
- Recipe data files (`recipes_for_embeddings.jsonl`, `full_format_recipes.json`)
- Pre-built FAISS index (`recipe_index.faiss`)

### Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd /path/to/meal_planner/culinary_agent_backend
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

## Usage

### Running the CLI

From the `culinary_agent_backend/src` directory:

```bash
cd /path/to/meal_planner/culinary_agent_backend/src
python main.py
```

Or as a module from the project root:

```bash
cd /path/to/meal_planner
python -m culinary_agent_backend.src.main
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

### Menu Options

1. **Generate Full Day Plan** - Creates a complete meal plan with breakfast, lunch, and dinner
2. **Modify ONLY Breakfast** - Updates breakfast while keeping lunch and dinner from previous plan
3. **Modify ONLY Lunch** - Updates lunch while keeping breakfast and dinner from previous plan
4. **Modify ONLY Dinner** - Updates dinner while keeping breakfast and lunch from previous plan
5. **Clear Session** - Removes saved state and starts fresh
6. **Quit** - Exits the application (saves current state)

### Building the FAISS Index

If `recipe_index.faiss` doesn't exist:
meal plan management with selective meal updates
- Menu-driven interface for intuitive meal planning
```bash
python -m culinary_agent_backend.src.indexing.build_index
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
pytest tests/
```

### Adding New Tools

1. Create a new file in `src/tools/`
2. Inherit from `smolagents.Tool`
3. Define `name`, `description`, `inputs`, and `output_type`
4. Implement `forward(self, **kwargs)` method
5. Register in `agent/core.py` `_initialize_tools()`

## Original Notebooks

The production code was refactored from:
- [`build_index.ipynb`](../build_index.ipynb) → `src/indexing/build_index.py`
- [`agentic_rag.ipynb`](../agentic_rag.ipynb) → `src/agent/`, `src/tools/`, `src/utils/`

**Non-destructive refactoring**: Original notebooks remain untouched for reference.

## License

This project is licensed under the MIT License.