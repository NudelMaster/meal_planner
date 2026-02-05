# Culinary Demo – Corrective RAG Agent

A retrieval-augmented generation (RAG) system for meal planning that demonstrates multi-stage retrieval with automatic fallback. The agent queries a local vector database for recipes, evaluates relevance using an LLM, and falls back to web search when retrieval quality is insufficient.

## Problem Statement

Traditional recipe search relies on keyword matching or requires manual filtering. This system uses semantic search and LLM-based evaluation to retrieve contextually relevant recipes, with a fallback mechanism to ensure result quality even when local data is incomplete.

## System Architecture

The pipeline implements a **corrective RAG workflow** with intent-aware filtering:

1. **Query Optimization** – LLM expands user query into keyword-rich search terms (e.g., "high protein" → "chicken beans tofu fish")
2. **Intent Analysis** – LLM extracts requirements, restrictions, and evaluation focus from user request
3. **Semantic Retrieval** – Google embeddings convert optimized query to vector; Pinecone returns top-k similar recipes (top-30 candidates)
4. **Relevance Judging** – LLM evaluates candidates against intent analysis, selecting only recipes that match requirements and avoid restrictions
5. **Fallback Logic** – If no matches found or user requests more, trigger Tavily web search
6. **Recipe Formatting & Adaptation** – LLM formats selected recipes and generates adaptation options based on user goals
7. **Streamlit UI** – Interactive chat interface with recipe selection, adaptation, and session management

### Why This Approach

- **Intent-aware filtering** improves precision over naive similarity search
- **Multi-stage LLM calls** (optimize → analyze → judge) enable context-specific evaluation
- **Hybrid retrieval** (local DB + web) balances speed and coverage
- **Recipe adaptation** allows iterative refinement (e.g., "make it vegetarian" → generates 3 options)
- **Production-grade stack** (Pinecone, Cerebras, Tavily) demonstrates real-world API integration

## Tech Stack

| Component | Technology | Role |
|-----------|-----------|------|
| **Embeddings** | Google Generative AI (Gemini `text-embedding-004`) | Semantic search encoding |
| **Vector DB** | Pinecone (Serverless) | Recipe index and retrieval |
| **LLM** | Cerebras Llama 3.3-70B | Query optimization, intent analysis, relevance judging, formatting |
| **Web Search** | Tavily API | Fallback retrieval |
| **Frontend** | Streamlit | Chat interface |
| **Orchestration** | LlamaIndex Workflows | Multi-step agent pipeline |

## Key Features

- **Intent-driven retrieval**: LLM extracts requirements/restrictions before judging recipes
- **Smart candidate excerpting**: Only sends relevant recipe sections (ingredients, directions, time) to judge based on intent
- **Multi-stage LLM calls**: Query optimizer → Intent analyzer → Relevance judge → Formatter
- **Recipe adaptation**: Select a recipe, specify a goal (e.g., "add more spice"), get 3 adaptation options
- **Session management**: Save/restore search history with adaptation tracking
- **Web fallback**: Tavily search triggers when local results are insufficient
- **Excluded titles tracking**: Prevents duplicate results across searches

## Project Structure

```
culinary-demo/
├── workflow.py         # CorrectiveRAGWorkflow (optimize → retrieve → judge → decide)
├── app.py              # Streamlit UI with recipe selection and adaptation
├── ingest.py           # One-time script: load recipes into Pinecone
├── start_server.py     # Streamlit app entry point
├── requirements.txt    # Python dependencies
└── README.md
```

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_key
CEREBRAS_API_KEY=your_cerebras_key
TAVILY_API_KEY=your_tavily_key
```

### 3. Ingest Recipe Data
Populate the Pinecone index (run once):
```bash
python ingest.py
```
**Note**: Ingestion loads recipes from `../culinary_agent/data/recipes/` (requires recipe dataset)

## Running the Demo

```bash
python start_server.py
```

Access the chat interface at **http://localhost:8501**

## Example Interaction

### Basic Search
**User**: "I need a high-protein breakfast under 400 calories"

**System**:
1. **Query Optimizer**: "high-protein breakfast under 400 calories" → "chicken eggs tofu beans breakfast protein low-calorie"
2. **Intent Analyzer**: Extracts requirements (high protein, <400 cal) and evaluation focus (ingredients, nutritional info)
3. **Pinecone Retrieval**: Returns 30 candidate recipes via semantic search
4. **Relevance Judge**: Evaluates candidates against requirements, selects top matches
5. **Response**: Returns JSON array of matching recipes with title, full text, and match reason

### Adaptation Workflow
**User**: Selects "Greek Yogurt Parfait" → "Make it vegan"

**System**:
1. Sends current recipe + adaptation goal to LLM
2. Returns 3 options:
   - **Option 1**: Ingredient swaps (coconut yogurt, maple syrup instead of honey)
   - **Option 2**: Add-ons (chia seeds, protein powder)
   - **Option 3**: New vegan recipe aligned with original request
3. User selects an option → becomes new "current recipe" (can adapt again)

### Web Fallback
**User**: "Show me traditional Uzbek plov with lamb"

**System**:
1. Pinecone returns limited results (judge selects 0 recipes)
2. User clicks "Search Online (Tavily)"
3. Tavily searches web → LLM extracts recipes from results
4. Judge filters web results using same intent-analysis logic
5. Returns web-sourced recipes with `🌐` indicator

## Workflow Details

### Step 1: `optimize_query`
- Input: User query string
- LLM Call: Query optimizer prompt
- Output: Keyword-rich search query OR web search event

### Step 2a: `retrieve` (DB mode)
- Input: Optimized query
- Action: Pinecone similarity search (top-30)
- Output: Retrieved nodes

### Step 2b: `web_search` (Web mode)
- Input: Optimized query
- Action: Tavily API call → LLM formats results
- Output: Web recipes (with intent-based judging)

### Step 3: `eval_relevance`
- Input: Retrieved nodes
- LLM Calls:
  1. Intent analyzer (extracts requirements/restrictions)
  2. Relevance judge (selects matching recipes)
- Output: JSON array of selected recipes

### Step 4: `decide`
- Input: Selected recipes JSON
- Output: StopEvent (returned to Streamlit)

## Future Improvements

- **Caching layer**: Redis for repeat queries to reduce API costs
- **User feedback loop**: Store thumbs-up/down to retrain retrieval ranking
- **Nutritional metadata**: Add structured filters (calories, macros) to vector metadata
- **Recipe generation**: Fine-tune LLM to create new recipes from ingredient lists
- **Evaluation metrics**: Track precision@k, judge agreement, and fallback rate over time
- **Batch embeddings**: Parallelize embedding generation during ingestion (currently sequential)

## API Usage & Rate Limits

Monitor usage via provider dashboards to stay within plan limits:

- **Google AI Studio (Embeddings)**: Model-specific limits (requests/min, tokens/min).  
  [Docs](https://ai.google.dev/gemini-api/docs/rate-limits) | [Dashboard](https://aistudio.google.com)

- **Pinecone (Vector DB)**: Serverless limits – 100 req/s per namespace, 2,000 query read units/s per index.  
  [Docs](https://docs.pinecone.io/docs/limits) | [Dashboard](https://app.pinecone.io)

- **Cerebras (LLM Inference)**: Account/plan-specific quotas. **Note**: This project uses Llama 3.3-70B (not 3.1-70B as originally stated).  
  [Docs](https://docs.cerebras.ai) | [Dashboard](https://cloud.cerebras.ai)

- **Tavily (Web Search)**: 100 RPM (dev keys), 1,000 RPM (production keys).  
  [Docs](https://docs.tavily.com/documentation/rate-limits) | [Dashboard](https://app.tavily.com)

---

**Built by**: [NudelMaster](https://github.com/NudelMaster)  
**License**: MIT