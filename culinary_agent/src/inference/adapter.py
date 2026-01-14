"""Recipe adaptation tool using LLM."""

from typing import Any

from smolagents import Tool

from src.utils.decorators import robust_llm_call


class RecipeAdapterTool(Tool):
    """Rewrites a recipe to comply with a specific dietary constraint."""
    
    name = "adapt_recipe"
    description = "Rewrites a recipe to comply with a specific dietary constraint (e.g., 'Make this vegan')."
    inputs = {
        "recipe_text": {"type": "string", "description": "The original recipe text."},
        "target_diet": {"type": "string", "description": "The target diet (e.g., 'vegan', 'gluten-free')."}
    }
    output_type = "string"

    def __init__(self, model_engine: Any, **kwargs):
        """Initialize the recipe adapter tool.
        
        Args:
            model_engine: The LLM model engine to use for adaptation
        """
        super().__init__(**kwargs)
        self.model_engine = model_engine
    
    @robust_llm_call
    def forward(self, recipe_text: str, target_diet: str) -> str:
        """Adapt a recipe to match dietary constraints.
        
        Args:
            recipe_text: The original recipe text
            target_diet: The target dietary constraint (e.g., 'vegan', 'gluten-free')
            
        Returns:
            Adapted recipe text
        """
        # Construct a prompt for the LLM to do the rewriting
        prompt = f"""
        You are an expert chef. Rewrite the following recipe to be strictly {target_diet}.
        
        Rules:
        1. Replace ONLY forbidden ingredients with best culinary substitutes.
        2. Keep the original formatting.
        3. Do not change the dish identity (e.g., 'Beef Stew' becomes 'Lentil Stew', not 'Salad').
        
        Original Recipe:
        {recipe_text}
        
        Rewritten Recipe:
        """
        
        # Call the LLM (using smolagents' model wrapper)
        messages = [{"role": "user", "content": prompt}]
        
        # Fixed to return content instead of whole message for validation
        response = self.model_engine(messages)
        if hasattr(response, "content"):
            return response.content
        else:
            return str(response)
