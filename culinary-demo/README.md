# Culinary Demo – Corrective RAG Agent

A retrieval-augmented generation (RAG) system for meal planning that demonstrates multi-stage retrieval with automatic fallback. The agent queries a local vector database for recipes, evaluates relevance using an LLM, and falls back to web search when retrieval quality is insufficient.

## Problem Statement

Traditional recipe search relies on keyword matching or requires manual filtering. This system uses semantic search and LLM-based evaluation to retrieve contextually relevant recipes, with a fallback mechanism to ensure result quality even when local data is incomplete.

## System Architecture

The pipeline implements a **corrective RAG workflow**:

1. **User Query** – Natural language input (e.g., "high-protein vegetarian dinner")
2. **Semantic Retrieval** – Google embeddings convert query to vector; Pinecone returns top-k similar recipes
3. **Relevance Evaluation** – Cerebras Llama 3.1-70B grades retrieval quality
4. **Fallback Logic** – If relevance is low, trigger Tavily web search
5. **Response Generation** – LLM synthesizes final answer from retrieved + web results
6. **Streamlit UI** – Interactive chat interface with streaming responses

### Why This Approach

- **Corrective mechanism** improves reliability over naive RAG
- **Hybrid retrieval** (local DB + web) balances speed and coverage
- **LLM evaluation** automates quality assessment at runtime
- **Production-grade stack** (Pinecone, Cerebras, Tavily) demonstrates real-world API integration

## Tech Stack

| Component | Technology | Role |
|-----------|-----------|------|
| **Embeddings** | Google Generative AI (Gemini) | Semantic search encoding |
| **Vector DB** | Pinecone (Serverless) | Recipe index and retrieval |
| **LLM** | Cerebras Llama 3.1-70B | Grading, generation |
| **Web Search** | Tavily API | Fallback retrieval |
| **Frontend** | Streamlit | Chat interface |
| **Orchestration** | LangGraph | Multi-step agent workflow |

## Key Features

- **Multi-stage retrieval**: Local vector search → relevance check → conditional web fallback
- **LLM-as-a-judge**: Automated quality scoring to trigger corrective actions
- **Streaming responses**: Real-time output from LLM to UI
- **Modular pipeline**: Easy to swap embedding models, vector stores, or LLMs
- **Rate limit aware**: Handles API constraints across multiple providers (see below)

## Project Structure

```
culinary-demo/
├── ingest.py           # One-time script: load recipes into Pinecone
├── start_server.py     # Streamlit app entry point
├── requirements.txt    # Python dependencies
├── .env.example        # Template for API keys
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

## Running the Demo

```bash
python start_server.py
```

Access the chat interface at **http://localhost:8501**

## Example Interaction

**User**: "I need a high-protein breakfast under 400 calories"

**System**:
1. Retrieves 5 recipes from Pinecone via semantic search
2. LLM evaluates: "Relevance score: 8/10 – results match criteria"
3. Returns recipes with protein content, calorie counts, and preparation steps

**User**: "Show me traditional Uzbek dishes with lamb"

**System**:
1. Pinecone returns limited results (score: 3/10)
2. Triggers Tavily web search for broader coverage
3. Combines local + web results in final response

## Future Improvements

- **User feedback loop**: Store thumbs-up/down to retrain retrieval ranking
- **Dietary constraints**: Add structured filters (vegan, gluten-free, etc.) to vector metadata
- **Recipe generation**: Fine-tune LLM to create new recipes from ingredient lists
- **Caching layer**: Redis for repeat queries to reduce API costs
- **Evaluation metrics**: Track precision@k and fallback rate over time

## API Usage & Rate Limits

Monitor usage via provider dashboards to stay within plan limits:

- **Google AI Studio (Embeddings)**: Model-specific limits (requests/min, tokens/min).  
  [Docs](https://ai.google.dev/gemini-api/docs/rate-limits) | [Dashboard](https://aistudio.google.com)

- **Pinecone (Vector DB)**: Serverless limits – 100 req/s per namespace, 2,000 query read units/s per index.  
  [Docs](https://docs.pinecone.io/docs/limits) | [Dashboard](https://app.pinecone.io)

- **Cerebras (LLM Inference)**: Account/plan-specific quotas.  
  [Docs](https://docs.cerebras.ai) | [Dashboard](https://cloud.cerebras.ai)

- **Tavily (Web Search)**: 100 RPM (dev keys), 1,000 RPM (production keys).  
  [Docs](https://docs.tavily.com/documentation/rate-limits) | [Dashboard](https://app.tavily.com)

---

**Built by**: [NudelMaster](https://github.com/NudelMaster)  
**License**: MIT