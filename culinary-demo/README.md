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
