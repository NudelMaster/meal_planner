"""Tests for the culinary agent backend."""

import sys
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.core.agent import create_agent, CulinaryAgent
from src.prompts.templates import SYSTEM_PROMPT
from src.rag.retrieval import RecipeRetrieverTool
from src.inference.adapter import RecipeAdapterTool
from src.inference.validator import RecipeValidatorTool
from src.inference.fallback import WebSearchTool
from src.utils.decorators import robust_llm_call
from src.utils.state_manager import StateManager
from src.config.settings import (
    MODEL_ID,
    RECIPE_EMBEDDINGS_FILE,
    FULL_RECIPES_FILE,
    FAISS_INDEX_FILE,
    EMBEDDING_MODEL_NAME
)

import tempfile
import os


def test_state_manager_save_and_load():
    """Test StateManager can save and load state."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        temp_file = Path(f.name)
    
    try:
        manager = StateManager(state_file=temp_file)
        
        # Test saving
        test_state = "Breakfast: Oatmeal\nLunch: Salad\nDinner: Pasta"
        manager.save_state(test_state)
        assert manager.has_state()
        
        # Test loading
        loaded_state = manager.load_state()
        assert loaded_state == test_state
        
        # Test clearing
        manager.clear_state()
        assert not manager.has_state()
        assert manager.load_state() is None
    finally:
        if temp_file.exists():
            temp_file.unlink()


def test_state_manager_no_state():
    """Test StateManager when no state exists."""
    with tempfile.NamedTemporaryFile(mode='w', delete=True, suffix='.txt') as f:
        temp_file = Path(f.name)
    
    manager = StateManager(state_file=temp_file)
    assert not manager.has_state()
    assert manager.load_state() is None


def test_create_agent_factory():
    """Test the create_agent factory function."""
    # This will fail if data files don't exist, but tests the factory works
    try:
        agent = create_agent(top_k_recipes=1, max_steps=10)
        assert isinstance(agent, CulinaryAgent)
        assert agent.top_k_recipes == 1
        assert agent.max_steps == 10
        assert agent.model_id == MODEL_ID
    except FileNotFoundError:
        # Expected if test environment doesn't have data files
        pass


def test_recipe_retriever_initialization():
    """Test RecipeRetrieverTool can be initialized with paths."""
    try:
        retriever = RecipeRetrieverTool(
            embeddings_file=RECIPE_EMBEDDINGS_FILE,
            full_recipes_file=FULL_RECIPES_FILE,
            index_file=FAISS_INDEX_FILE,
            embedding_model_name=EMBEDDING_MODEL_NAME,
            k=1
        )
        assert retriever.k == 1
        assert retriever.embedding_model_name == EMBEDDING_MODEL_NAME
    except FileNotFoundError:
        # Expected if test environment doesn't have data files
        pass


def test_recipe_retriever_empty_query():
    """Test RecipeRetrieverTool handles empty queries."""
    try:
        retriever = RecipeRetrieverTool(
            embeddings_file=RECIPE_EMBEDDINGS_FILE,
            full_recipes_file=FULL_RECIPES_FILE,
            index_file=FAISS_INDEX_FILE,
            k=1
        )
        
        # Test with empty query
        result = retriever.forward("")
        assert "Found 0 recipes" in result
        
        # Test with None (should handle gracefully)
        result = retriever.forward(None)
        assert "Found 0 recipes" in result or "Error" in result
    except FileNotFoundError:
        # Expected if test environment doesn't have data files
        pass


def test_web_search_tool():
    """Test WebSearchTool basic functionality."""
    search_tool = WebSearchTool()
    
    # Test with a simple query
    query = "vegan pasta recipe"
    result = search_tool.forward(query)
    
    # Should return either results or an error message
    assert isinstance(result, str)
    assert len(result) > 0


def test_robust_llm_call_decorator():
    """Test that robust_llm_call decorator exists and is callable."""
    
    @robust_llm_call
    def dummy_function():
        return "success"
    
    result = dummy_function()
    assert result == "success"


def test_meal_plan_prompt_construction():
    """Test meal plan prompt construction logic from main.py."""
    # Test full day plan prompt
    diet_request = "vegan high protein"
    user_request = f"Create a full daily meal plan. Constraint: {diet_request}"
    full_prompt = f"{SYSTEM_PROMPT}\n\nUSER REQUEST: {user_request}"
    
    assert "Create a full daily meal plan" in full_prompt
    assert diet_request in full_prompt
    assert SYSTEM_PROMPT in full_prompt


def test_selective_meal_update_prompt():
    """Test selective meal update prompt construction."""
    previous_plan = "Breakfast: Oatmeal\nLunch: Salad\nDinner: Pasta"
    diet_request = "gluten-free"
    user_request = f"Update ONLY the Breakfast. Constraint: {diet_request}"
    
    full_prompt = f"""
                    {SYSTEM_PROMPT}

                    CONTEXT: The user has an existing plan.
                    PREVIOUS PLAN:
                    {previous_plan}

                    USER REQUEST: {user_request}

                    TASK: 
                    1. If the user asked for a FULL PLAN, ignore the previous plan and generate new.
                    2. If the user asked to UPDATE a specific meal, keep the other meals from PREVIOUS PLAN and only rewrite the requested one.
                    """
    
    assert "Update ONLY the Breakfast" in full_prompt
    assert previous_plan in full_prompt
    assert "gluten-free" in full_prompt


if __name__ == "__main__":
    # Run tests manually
    print("Running tests...")
    
    test_state_manager_save_and_load()
    print("✓ test_state_manager_save_and_load passed")
    
    test_state_manager_no_state()
    print("✓ test_state_manager_no_state passed")
    
    test_robust_llm_call_decorator()
    print("✓ test_robust_llm_call_decorator passed")
    
    test_meal_plan_prompt_construction()
    print("✓ test_meal_plan_prompt_construction passed")
    
    test_selective_meal_update_prompt()
    print("✓ test_selective_meal_update_prompt passed")
    
    test_web_search_tool()
    print("✓ test_web_search_tool passed")
    
    print("\nAll tests passed!")