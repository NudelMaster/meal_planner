"""Recipe validation tool using LLM."""

from typing import Any

from smolagents import Tool

from utils.decorators import robust_llm_call


class RecipeValidatorTool(Tool):
    """Checks recipe compliance with dietary constraints."""
    
    name = "validate_recipe"
    description = "Checks recipe compliance. Returns strictly 'PASS' or 'FAIL'."
    inputs = {
        "recipe_text": {"type": "string", "description": "The recipe text."},
        "constraint": {"type": "string", "description": "The diet (e.g., 'vegan')."}
    }
    output_type = "string"

    def __init__(self, model_engine: Any, **kwargs):
        """Initialize the recipe validator tool.
        
        Args:
            model_engine: The LLM model engine to use for validation
        """
        super().__init__(**kwargs)
        self.model_engine = model_engine

    @robust_llm_call
    def forward(self, recipe_text: str, constraint: str) -> str:
        """Validate a recipe against dietary constraints.
        
        Args:
            recipe_text: The recipe text to validate
            constraint: The dietary constraint to check against
            
        Returns:
            Either 'PASS' or 'FAIL'
        """
        prompt = f"""
        Review this recipe for the strict constraint: "{constraint}".
        RECIPE: {recipe_text[:3000]}  # Truncate to avoid context errors
        
        If ANY forbidden ingredient is present, output FAIL.
        If it is safe, output PASS.
        
        Final Answer (Strictly 'PASS' or 'FAIL'):
        """
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.model_engine(messages)
            content = response.content if hasattr(response, "content") else str(response)
            
            # Deterministic Parsing
            if "FAIL" in content.upper():
                return "FAIL"
            return "PASS"
        except Exception:
            # Fallback to FAIL on model error to be safe
            return "FAIL"