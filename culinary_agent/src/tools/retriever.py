"""Recipe retrieval tool using FAISS vector search."""

import json
from pathlib import Path
from typing import Dict, List, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from smolagents import Tool

from utils.decorators import robust_llm_call


class RecipeRetrieverTool(Tool):
    """Retrieves the best matching recipes from the database using semantic search."""
    
    name = "retrieve_recipe"
    description = "Retrieves the best matching recipe/s from the database. Returns raw title and ingredients."
    inputs = {
        "query": {"type": "string", "description": "Dish name or ingredients."},
    }
    output_type = "string"
    
    def __init__(
        self,
        embeddings_file: Path,
        full_recipes_file: Path,
        index_file: Path,
        embedding_model_name: str = "BAAI/bge-m3",
        k: int = 1,
        **kwargs
    ):
        """Initialize the recipe retrieval tool.
        
        Args:
            embeddings_file: Path to the JSONL file with recipe embeddings metadata
            full_recipes_file: Path to the JSON file with full recipe details
            index_file: Path to the FAISS index file
            embedding_model_name: Name of the sentence transformer model
            k: Number of top recipes to retrieve (default: 1)
        """
        super().__init__(**kwargs)
        self.k = k
        self.embeddings_file = embeddings_file
        self.full_recipes_file = full_recipes_file
        self.index_file = index_file
        self.embedding_model_name = embedding_model_name
        
        print(f"Loading recipe retrieval system with top {k} results")
        
        # Load metadata from embeddings file
        self.metadata: List[Dict[str, Any]] = []
        with open(self.embeddings_file, 'r') as f:
            for line in f:
                self.metadata.append(json.loads(line))
        
        # Load full recipe details (with ingredients and directions)
        with open(self.full_recipes_file, 'r') as f:
            full_recipes = json.load(f)
        
        # HashTable for optimized lookup
        self.recipe_lookup: Dict[str, Dict[str, Any]] = {
            r.get('title', '').strip(): r 
            for r in full_recipes 
            if r.get('title')
        }
        
        # Load embedding model
        print(f"Loading embedding model {self.embedding_model_name}...")
        self.embed_model = SentenceTransformer(self.embedding_model_name)
        
        # Load FAISS index
        print("Loading FAISS index...")
        self.index = faiss.read_index(str(self.index_file))
        
        print(f"✓ Recipe retrieval system loaded: {len(self.metadata)} recipes indexed")
    
    @robust_llm_call
    def forward(self, query: str) -> str:
        """Search for recipes matching the query.
        
        Args:
            query: Natural language search query
            
        Returns:
            Formatted string with recipe titles, ingredients, and directions
        """
        try:
            if not query or not isinstance(query, str):
                return "Found 0 recipes."
            
            # 1. Embed the Query and Ensure Float 32 (Required by FAISS)
            query_vec = self.embed_model.encode([query], convert_to_tensor=False)
            query_vec = np.array(query_vec).astype('float32')
            
            # 2. Normalize euclidean distance
            faiss.normalize_L2(query_vec)
            
            # 3. Search
            _, indices = self.index.search(query_vec, self.k)
            
            # 4. Retrieve
            retrieved_docs = [
                self.metadata[idx] 
                for idx in indices[0] 
                if idx != -1
            ]
            
            # 5. Format Output
            output = f"Found {len(retrieved_docs)} recipes matching '{query}':\n\n"
            
            for i, doc in enumerate(retrieved_docs, 1):
                title = doc['title'].strip()
                full_recipe = self.recipe_lookup.get(title)
                
                output += f"{'='*40} Recipe {i} {'='*40}\n"
                output += f"TITLE: {title}\n"
                
                if full_recipe:
                    ingredients = full_recipe.get('ingredients', [])
                    output += "INGREDIENTS:\n" + "\n".join([f" - {ing}" for ing in ingredients])
                    
                    directions = full_recipe.get('directions', [])
                    output += "\n\nDIRECTIONS:\n" + "\n".join(
                        [f" {j}. {step}" for j, step in enumerate(directions, 1)]
                    )
                else:
                    # Fallback
                    output += f"SUMMARY: {doc.get('text_for_embedding', 'No details available')}"
                
                output += "\n\n"
                
            return output
            
        except Exception as e:
            return f"Found 0 recipes. Error during retrieval: {str(e)}"