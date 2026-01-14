"""Configuration module for the Culinary Agent."""

from .settings import (
    MODEL_ID,
    EMBEDDING_MODEL_NAME,
    RECIPE_EMBEDDINGS_FILE,
    FULL_RECIPES_FILE,
    FAISS_INDEX_FILE,
    PROJECT_ROOT,
    DATA_DIR,
    INDICES_DIR
)

__all__ = [
    'MODEL_ID',
    'EMBEDDING_MODEL_NAME',
    'RECIPE_EMBEDDINGS_FILE',
    'FULL_RECIPES_FILE',
    'FAISS_INDEX_FILE',
    'PROJECT_ROOT',
    'DATA_DIR',
    'INDICES_DIR'
]
