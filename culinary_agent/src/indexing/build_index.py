"""FAISS index builder for recipe embeddings."""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Ensure we can find src modules if running this script directly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class RecipeIndexBuilder:
    """Builds and manages FAISS index for recipe embeddings."""
    
    def __init__(
        self,
        embeddings_file: Path,
        index_file: Path,
        # UPDATED: Default to the lightweight model matching settings.py
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """Initialize the index builder.
        
        Args:
            embeddings_file: Path to the JSONL file with recipe data
            index_file: Path where the FAISS index will be saved
            embedding_model_name: Name of the sentence transformer model
        """
        self.embeddings_file = embeddings_file
        self.index_file = index_file
        self.embedding_model_name = embedding_model_name
    
    def load_data(self) -> List[Dict[str, Any]]:
        """Load recipe data from JSONL file."""
        data: List[Dict[str, Any]] = []
        print(f"Loading data from '{self.embeddings_file}'...")
        
        if not self.embeddings_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.embeddings_file}")
        
        with open(self.embeddings_file, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        print(f"Loaded {len(data)} recipes")
        if data:
            print(f"Example document keys: {list(data[0].keys())}")
        
        return data
    
    def generate_embeddings(self, documents: List[str]) -> np.ndarray:
        """Generate embeddings for documents."""
        print(f"Initializing embedding model: {self.embedding_model_name}...")
        embed_model = SentenceTransformer(self.embedding_model_name)
        
        print("Generating embeddings...")
        recipe_embeddings = embed_model.encode(
            documents,
            convert_to_tensor=False,
            show_progress_bar=True
        )
        recipe_embeddings = np.array(recipe_embeddings).astype('float32')
        print(f"Generated embeddings shape: {recipe_embeddings.shape}")
        
        return recipe_embeddings
    
    def build_and_save_index(self, embeddings: np.ndarray) -> None:
        """Build FAISS index and save to disk."""
        print("Normalizing vectors...")
        # Normalize with respect to euclidean norm
        faiss.normalize_L2(embeddings)
        
        print("Creating FAISS index...")
        # Create FAISS Index
        dimension = embeddings.shape[1]
        # Using IndexFlatIP for cosine similarity (on normalized vectors)
        index = faiss.IndexFlatIP(dimension)
        
        index.add(embeddings)
        print(f"Indexed {index.ntotal} documents with Cosine Similarity.")
        
        # Save the Index
        print(f"Saving FAISS index to '{self.index_file}'...")
        if not self.index_file.parent.exists():
             self.index_file.parent.mkdir(parents=True, exist_ok=True)
             
        faiss.write_index(index, str(self.index_file))
        print("✓ Index saved successfully")
    
    def build(self) -> None:
        """Complete build pipeline: load data, generate embeddings, build index."""
        # 1. Load the Data
        data = self.load_data()
        
        # 2. Extract text for embedding
        documents = [d['text_for_embedding'] for d in data]
        
        # 3. Generate Embeddings
        embeddings = self.generate_embeddings(documents)
        
        # 4. Build and Save Index
        self.build_and_save_index(embeddings)
        
        print("\n" + "="*50)
        print("✓ Index building complete!")
        print("="*50)


def main():
    """Entry point for building the index."""
    # UPDATED: Import from src.config.settings
    try:
        from src.config.settings import (
            RECIPE_EMBEDDINGS_FILE,
            FAISS_INDEX_FILE,
            EMBEDDING_MODEL_NAME
        )
    except ImportError:
        # Fallback if running from root without src installed as package
        try:
            from config.settings import (
                RECIPE_EMBEDDINGS_FILE,
                FAISS_INDEX_FILE,
                EMBEDDING_MODEL_NAME
            )
        except ImportError as e:
            print(f"CRITICAL ERROR: Could not import settings. {e}")
            return

    print(f"Target Model: {EMBEDDING_MODEL_NAME}")
    
    builder = RecipeIndexBuilder(
        embeddings_file=RECIPE_EMBEDDINGS_FILE,
        index_file=FAISS_INDEX_FILE,
        embedding_model_name=EMBEDDING_MODEL_NAME
    )
    
    builder.build()


if __name__ == "__main__":
    main()