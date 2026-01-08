from culinary_agent_backend.src.tools.retriever import RecipeRetrieverTool
from culinary_agent_backend.src.tools.adapter import RecipeAdapterTool, RecipeValidatorTool
from culinary_agent_backend.src.tools.web_search import WebSearchTool

def test_recipe_retriever():
    retriever = RecipeRetrieverTool(k=3)
    result = retriever.forward("chicken curry")
    assert "Found" in result

def test_recipe_adapter():
    adapter = RecipeAdapterTool(model_engine=None)  # Mock or pass a real model engine
    original_recipe = "Chicken Curry: Ingredients: chicken, curry powder."
    adapted_recipe = adapter.forward(original_recipe, "vegan")
    assert "chicken" not in adapted_recipe

def test_recipe_validator():
    validator = RecipeValidatorTool(model_engine=None)  # Mock or pass a real model engine
    recipe = "Chicken Curry: Ingredients: chicken, curry powder."
    status = validator.forward(recipe, "vegan")
    assert status == "FAIL"

def test_web_search_tool():
    search_tool = WebSearchTool()
    result = search_tool.forward("chicken curry recipe")
    assert "No recipes found" not in result