import json
import os
import sys
import time
import warnings
from pathlib import Path
# Fix: Filter out Google's gRPC warnings
warnings.filterwarnings("ignore", category=FutureWarning, module=r"google\..*")

from dotenv.main import load_dotenv
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from embed_device import configure_cuda_before_torch, select_embed_device

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
EMBEDDINGS_PATH = DATA_DIR / "recipes_for_embeddings.jsonl"
FULL_RECIPES_PATH = DATA_DIR / "full_format_recipes.json"

INDEX_NAME = "culinary-demo"
EMBED_MODEL = "google/embeddinggemma-300m"
EMBED_DIM = 768
# EmbeddingGemma uses asymmetric prompts: documents and queries get different
# instruction prefixes. These MUST stay identical to app.py or retrieval breaks.
EMBED_QUERY_INSTRUCTION = "task: search result | query: "
EMBED_TEXT_INSTRUCTION = "title: none | text: "
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"


def _require_env(name):
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _load_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_full_recipes(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_documents(embedding_records, full_recipes):
    full_lookup = {
        recipe.get("title", "").strip(): recipe
        for recipe in full_recipes
        if recipe.get("title")
    }

    documents = []
    for idx, record in enumerate(embedding_records):
        title = (record.get("title") or "").strip()
        
        full_recipe = full_lookup.get(title, {})
        if not full_recipe:
            continue
            
        ingredients = full_recipe.get("ingredients", [])
        directions = full_recipe.get("directions", [])
        categories = full_recipe.get("categories", [])
        
        # Construct the RICH TEXT block
        full_text_lines = [f"Title: {title}"]
        
        if categories:
             full_text_lines.append(f"Categories: {', '.join(categories)}")
             
        if ingredients:
            full_text_lines.append("Ingredients:")
            full_text_lines.extend([f"- {item}" for item in ingredients])
            
        if directions:
            full_text_lines.append("Directions:")
            full_text_lines.extend([f"{step}" for step in directions])
        
        full_text = "\n".join(full_text_lines)
        
        # Minimal metadata
        metadata = {
            "title": title,
            "recipe_id": f"recipe-{idx}"
        }

        documents.append(
            Document(
                text=full_text,
                metadata=metadata,
                doc_id=f"recipe-{idx}",
            )
        )

    return documents


def _ensure_index(client, dimension):
    existing = client.list_indexes()
    if hasattr(existing, "names"):
        index_names = existing.names()
    else:
        index_names = [entry["name"] for entry in existing]

    if INDEX_NAME in index_names:
        return

    try:
        client.create_index(
            name=INDEX_NAME,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    except Exception as exc:
        if "already exists" not in str(exc).lower():
            raise


def main():
    load_dotenv()
    configure_cuda_before_torch()

    pinecone_api_key = _require_env("PINECONE_API_KEY")

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"Missing embeddings data: {EMBEDDINGS_PATH}")
    if not FULL_RECIPES_PATH.exists():
        raise FileNotFoundError(f"Missing full recipes data: {FULL_RECIPES_PATH}")
    # Initialize Embedding Model (local EmbeddingGemma — runs on-device, no API quota).
    embed_device = select_embed_device()
    embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL,
        query_instruction=EMBED_QUERY_INSTRUCTION,
        text_instruction=EMBED_TEXT_INSTRUCTION,
        embed_batch_size=256,
        device=embed_device,
    )

    # Probe dimension
    dimension = len(embed_model.get_text_embedding("test"))

    print("Loading data...")
    embedding_records = _load_jsonl(EMBEDDINGS_PATH)
    full_recipes = _load_full_recipes(FULL_RECIPES_PATH)
    
    print("Building documents (Rich Text Mode)...")
    documents = _build_documents(embedding_records, full_recipes)
    if not documents:
        raise RuntimeError("No documents found to ingest.")

    client = Pinecone(api_key=pinecone_api_key)
    _ensure_index(client, dimension)

    force = "--force" in sys.argv or os.getenv("FORCE_REINGEST", "").lower() in ("1", "true", "yes")

    index = client.Index(INDEX_NAME)
    stats = index.describe_index_stats()
    if stats.total_vector_count > 0:
        if not force:
            print(f"⚠️ Index '{INDEX_NAME}' already contains {stats.total_vector_count} vectors.")
            print(
                "Skipping ingestion to prevent overwriting. Re-run with --force to wipe and re-ingest "
                "(required after changing the embedding model, so query and document vectors come "
                "from the same model)."
            )
            return
        print(
            f"--force set: clearing {stats.total_vector_count} stale vectors from '{INDEX_NAME}' "
            f"before re-ingesting with {EMBED_MODEL}..."
        )
        index.delete(delete_all=True)
        # Bulk delete is eventually consistent; wait until the index reports empty.
        for _ in range(60):
            if index.describe_index_stats().total_vector_count == 0:
                break
            time.sleep(1)
    vector_store = PineconeVectorStore(pinecone_index=index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=50)

    # Embedding runs on GPU (embed_batch_size=256); insert_batch_size controls the
    # Pinecone upsert chunk size. Larger batches speed up the upload.
    print(f"Uploading {len(documents)} documents to Pinecone (Batch Size: 256)...")

    VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
        transformations=[text_splitter],
        # Optimization: Increase ingestion batch size (Pinecone can handle it)
        insert_batch_size=256
    )

    print(f"Uploaded {len(documents)} recipes to Pinecone index '{INDEX_NAME}'.")


if __name__ == "__main__":
    main()
