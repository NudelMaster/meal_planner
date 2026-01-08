from culinary_agent_backend.agent.core import CodeAgent
from culinary_agent_backend.tools.retriever import RecipeRetrieverTool
from culinary_agent_backend.tools.adapter import RecipeAdapterTool, RecipeValidatorTool
from culinary_agent_backend.tools.web_search import WebSearchTool
from culinary_agent_backend.utils.decorators import robust_llm_call

def test_recipe_retriever():
    retriever = RecipeRetrieverTool()
    query = "chicken curry"
    result = retriever.forward(query)
    assert "Found" in result

def test_recipe_adapter():
    adapter = RecipeAdapterTool(model_engine=CodeAgent)
    original_recipe = "Chicken Curry with rice"
    target_diet = "vegan"
    adapted_recipe = adapter.forward(original_recipe, target_diet)
    assert "vegan" in adapted_recipe.lower()

def test_recipe_validator():
    validator = RecipeValidatorTool(model_engine=CodeAgent)
    recipe_text = "Chicken Curry with rice"
    constraint = "vegan"
    status = validator.forward(recipe_text, constraint)
    assert status == "FAIL"

def test_web_search_tool():
    search_tool = WebSearchTool()
    query = "chicken curry recipe"
    result = search_tool.forward(query)
    assert "No recipes found" not in result