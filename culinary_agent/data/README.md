# Data Directory

This directory contains recipe data used by the Culinary Agent system.

## Files

### recipes/
- **recipes_for_embeddings.jsonl**: Recipe data in JSONL format for embedding generation
- **full_format_recipes.json**: Complete recipe database with full details

## Usage

These files are automatically loaded by:
- `src/indexing/build_index.py` - To build FAISS indices
- `src/tools/retriever.py` - For recipe lookup

## Data Format

### recipes_for_embeddings.jsonl
Each line is a JSON object:
```json
{
  "title": "Recipe Name",
  "ingredients": "ingredient1, ingredient2, ...",
  "directions": "Step 1. Step 2. ..."
}
```

### full_format_recipes.json
Array of recipe objects with complete information:
```json
[
  {
    "title": "Recipe Name",
    "ingredients": ["ingredient1", "ingredient2"],
    "directions": ["Step 1", "Step 2"],
    "nutrition": {...},
    "tags": [...]
  }
]
```

## Adding New Recipes

1. Add to both files following the formats above
2. Rebuild the FAISS index:
   ```bash
   cd /path/to/culinary_agent
   python -m src.indexing.build_index
   ```

## Data Sources

[Add information about where your recipe data comes from]

## License

[Add license information for your recipe data]
