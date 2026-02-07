# Meal Planner – Corrective RAG Agent

A corrective retrieval-augmented generation (RAG) system for meal planning that combines semantic search, LLM-based intent filtering, and web-search fallback to ensure relevant results.

The system implements a multi-stage pipeline: query optimization, intent analysis, semantic retrieval, relevance judging, and recipe adaptation. It integrates vector search, LLM reasoning, and an interactive Streamlit interface to provide an end-to-end ML-driven application.

**Key Features:**
- Intent-aware recipe retrieval using LlamaIndex workflows
- Multi-stage LLM pipeline (query optimization → intent analysis → relevance judging)
- Hybrid search (Pinecone vector DB + Tavily web search)
- Interactive recipe adaptation with iterative refinement
- Session management and search history

**Tech Stack:** Pinecone, Cerebras Llama , Google Embeddings, Tavily, Streamlit

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
