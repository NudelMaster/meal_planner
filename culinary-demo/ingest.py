import json
import os
import warnings
from pathlib import Path

# Fix: Filter out Google's gRPC warnings
warnings.filterwarnings("ignore", category=FutureWarning, module=r"google\..*")

from dotenv.main import load_dotenv
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline

# Fix: Updated to latest Google GenAI SDK recommendation (llama-index-embeddings-google is wrapper)
try:
    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
except ImportError:
    # If explicit import fails, rely on standard package resolution which might have fixed namespace
    try:
        from llama_index.embeddings.google import GoogleGenAIEmbedding
    except ImportError:
        from llama_index.embeddings.google import GeminiEmbedding as GoogleGenAIEmbedding

from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT.parent / "culinary_agent" / "data" / "recipes"
EMBEDDINGS_PATH = DATA_DIR / "recipes_for_embeddings.jsonl"
FULL_RECIPES_PATH = DATA_DIR / "full_format_recipes.json"

INDEX_NAME = "culinary-demo"
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

    google_api_key = _require_env("GOOGLE_API_KEY")
    pinecone_api_key = _require_env("PINECONE_API_KEY")

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"Missing embeddings data: {EMBEDDINGS_PATH}")
    if not FULL_RECIPES_PATH.exists():
        raise FileNotFoundError(f"Missing full recipes data: {FULL_RECIPES_PATH}")

    # Initialize Embedding Model
    try:
        from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
        embed_model = GoogleGenAIEmbedding(
            api_key=google_api_key,
            model_name="models/text-embedding-004",
            # Optimization: Try to increase batch size for embedding requests if supported
            embed_batch_size=100, 
        )
    except ImportError:
        try:
            from llama_index.embeddings.google import GoogleGenAIEmbedding
        except ImportError:
            from llama_index.embeddings.google import GeminiEmbedding as GoogleGenAIEmbedding
            
        embed_model = GoogleGenAIEmbedding(
            api_key=google_api_key,
            model_name="models/text-embedding-004",
            embed_batch_size=100,
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

    # LIMIT FOR DEMO SPEED: Only ingest first 2000 recipes
    # This allows for a fast (~10 min) run to verify everything works.
    # To ingest all, comment out this line.
    # print("limiting to 2000 recipes for speed...")
    # documents = documents[:2000]

    client = Pinecone(api_key=pinecone_api_key)
    _ensure_index(client, dimension)

    index = client.Index(INDEX_NAME)
    stats = index.describe_index_stats()
    if stats.total_vector_count > 0:
        print(f"⚠️ Index '{INDEX_NAME}' already contains {stats.total_vector_count} vectors.")
        print("Skipping ingestion to prevent overwriting. To force update, delete the index from Pinecone console.")
        return
    vector_store = PineconeVectorStore(pinecone_index=index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=50)

    # Optimization: Parallel processing with IngestionPipeline could help, 
    # but for now we increase batch size in the index creation.
    print(f"Uploading {len(documents)} documents to Pinecone (Batch Size: 256)...")
    
    # We use the IngestionPipeline explicitly to control workers if needed, 
    # but run_transformations within from_documents is usually single-threaded.
    # Increasing batch_size here helps with Pinecone upload speed.
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
