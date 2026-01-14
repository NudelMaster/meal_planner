#!/bin/bash
# Build FAISS index for recipe embeddings

set -e

echo "🔨 Building Recipe Index"
echo "========================"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the index builder
python3 -c "
from pathlib import Path
from src.rag.embedding import build_recipe_index
from src.config.settings import RECIPE_EMBEDDINGS_FILE, FAISS_INDEX_FILE, EMBEDDING_MODEL_NAME

print('Starting index build...')
build_recipe_index(
    embeddings_file=RECIPE_EMBEDDINGS_FILE,
    index_file=FAISS_INDEX_FILE,
    embedding_model_name=EMBEDDING_MODEL_NAME
)
print('✅ Index build complete!')
"

echo ""
echo "✅ Recipe index built successfully!"
echo "Index saved to: indices/recipe_index.faiss"
