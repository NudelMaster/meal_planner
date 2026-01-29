# Culinary Demo - Corrective RAG Agent

A meal planning assistant that retrieves recipes from a Pinecone database, evaluates their relevance using Cerebras (Llama 3.1-70b), and falls back to Tavily web search if needed.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**:
   Create a `.env` file with:
   - `GOOGLE_API_KEY` (Embeddings)
   - `PINECONE_API_KEY` (Vector DB)
   - `CEREBRAS_API_KEY` (LLM Inference)
   - `TAVILY_API_KEY` (Web Search)

## Usage

1. **Ingest Data** (One-time setup):
   Populates the Pinecone index with your recipe data.
   ```bash
   python ingest.py
   ```

2. **Run the Application**:
   Launches the Streamlit chat interface.
   ```bash
   python start_server.py
   ```
   Access the app at `http://localhost:8501`.

## API Usage & Rate Limits

Use the dashboards below to monitor usage and review current rate limits (limits can
vary by plan and model).

- **Google AI Studio (Embeddings via `GoogleGenAIEmbedding`)**: Gemini API limits are
  model-specific (requests/minute and tokens/minute). See rate limits docs at
  https://ai.google.dev/gemini-api/docs/rate-limits and usage at
  https://aistudio.google.com. This project uses llama for these embeddings
- **Pinecone (Vector DB)**: Serverless data-plane limits include 100 requests/second per
  namespace for query/upsert/delete/update and 2,000 query read units/second per index
  (Starter/Standard/Enterprise). Full limits by plan:
  https://docs.pinecone.io/docs/limits. Usage dashboard: https://app.pinecone.io.
- **Cerebras (LLM Inference)**: Rate limits and quotas are account/plan-specific. Check
  Cerebras Cloud docs and usage at https://docs.cerebras.ai and
  https://cloud.cerebras.ai.
- **Tavily (Web Search)**: Development keys are limited to 100 RPM and production keys
  to 1,000 RPM; crawl endpoint is 100 RPM for both. Details:
  https://docs.tavily.com/documentation/rate-limits. Usage dashboard:
  https://app.tavily.com.
