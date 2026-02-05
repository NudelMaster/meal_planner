# Meal Planner

A collection of AI-powered meal planning and recipe discovery projects.

## Projects

### [Culinary Demo – Corrective RAG Agent](./culinary-demo)

A production-grade retrieval-augmented generation (RAG) system for recipe search with multi-stage LLM evaluation and automatic web fallback.

**Key Features:**
- Intent-aware recipe retrieval using LlamaIndex workflows
- Multi-stage LLM pipeline (query optimization → intent analysis → relevance judging)
- Hybrid search (Pinecone vector DB + Tavily web search)
- Interactive recipe adaptation with iterative refinement
- Session management and search history

**Tech Stack:** Pinecone, Cerebras Llama 3.3-70B, Google Embeddings, Tavily, Streamlit

[View detailed documentation →](./culinary-demo/README.md)

## Repository Structure

```
meal_planner/
├── culinary-demo/          # Corrective RAG recipe assistant
│   ├── workflow.py         # LlamaIndex workflow implementation
│   ├── app.py              # Streamlit UI
│   ├── ingest.py           # Vector DB ingestion
│   └── README.md           # Full documentation
├── notebooks/              # Exploratory analysis and prototyping
└── README.md               # This file
```

## Quick Start

Navigate to a project directory and follow its README:

```bash
cd culinary-demo
pip install -r requirements.txt
python start_server.py
```

---

**Author**: [NudelMaster](https://github.com/NudelMaster)  
**License**: MIT