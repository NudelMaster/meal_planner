"""Core agent initialization and management."""

import os
import time
from pathlib import Path
from typing import List, Optional, Any

from smolagents import CodeAgent, InferenceClientModel
from dotenv import load_dotenv

from src.rag.retrieval import RecipeRetrieverTool
from src.inference.adapter import RecipeAdapterTool
from src.inference.validator import RecipeValidatorTool
from src.inference.fallback import WebSearchTool
from src.config.settings import (
    MODEL_ID,
    RECIPE_EMBEDDINGS_FILE,
    FULL_RECIPES_FILE,
    FAISS_INDEX_FILE,
    EMBEDDING_MODEL_NAME
)
from src.utils.decorators import robust_llm_call


class CulinaryAgent:
    """Manages the culinary agent and its tools."""
    
    def __init__(
        self,
        model_id: str = MODEL_ID,
        top_k_recipes: int = 1,
        max_steps: int = 12
    ):
        """Initialize the culinary agent.
        
        Args:
            model_id: HuggingFace model ID for the LLM
            top_k_recipes: Number of recipes to retrieve per query
            max_steps: Maximum reasoning steps for the agent
        """
        # Load environment variables
        load_dotenv()
        
        self.model_id = model_id
        self.top_k_recipes = top_k_recipes
        self.max_steps = max_steps
        
        print(f"Initializing Culinary Agent with model: {self.model_id}")
        
        # Initialize the LLM model
        self.model = InferenceClientModel(model_id=self.model_id)
        
        # Initialize tools
        self.tools = self._initialize_tools()
        
        # Initialize the agent
        self.agent = self._initialize_agent()
        
        print("✓ Culinary Agent ready!")
    
    def _initialize_tools(self) -> List[Any]:
        """Initialize all agent tools.
        
        Returns:
            List of initialized tool instances
        """
        print("Initializing tools...")
        
        # Recipe Retriever Tool
        retriever_tool = RecipeRetrieverTool(
            embeddings_file=RECIPE_EMBEDDINGS_FILE,
            full_recipes_file=FULL_RECIPES_FILE,
            index_file=FAISS_INDEX_FILE,
            embedding_model_name=EMBEDDING_MODEL_NAME,
            k=self.top_k_recipes
        )
        
        # Recipe Adapter Tool
        adapter_tool = RecipeAdapterTool(model_engine=self.model)
        
        # Recipe Validator Tool
        validator_tool = RecipeValidatorTool(model_engine=self.model)
        
        # Web Search Tool
        search_tool = WebSearchTool()
        
        print("✓ All tools initialized")
        
        return [retriever_tool, adapter_tool, validator_tool, search_tool]
    
    def _initialize_agent(self) -> CodeAgent:
        """Initialize the CodeAgent.
        
        Returns:
            Initialized CodeAgent instance
        """
        print("Initializing CodeAgent...")
        
        agent = CodeAgent(
            tools=self.tools,
            model=self.model,
            verbosity_level = 2,
            add_base_tools=True,  # Enable scratchpad for Python logic
            max_steps=self.max_steps
        )
        
        print("✓ CodeAgent initialized")
        
        return agent
    
    @robust_llm_call
    def run(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Run the agent with a user prompt.
        
        Args:
            prompt: User's request/query
            system_prompt: Optional system prompt to override default
            
        Returns:
            Agent's response
        """
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUSER REQUEST: {prompt}"
        
        return self.agent.run(full_prompt)
    
    def run_with_retry(self, prompt: str, max_retries: int = 5) -> str:
        """Run the agent with retry logic for server failures.
        
        Args:
            prompt: User's request/query
            max_retries: Maximum number of retry attempts
            
        Returns:
            Agent's response
        """
        for attempt in range(max_retries):
            try:
                return self.run(prompt)
            except Exception as e:
                error_msg = str(e)
                if "503" in error_msg or "Service Temporarily Unavailable" in error_msg or "404" in error_msg:
                    wait_time = 2 ** (attempt + 1)
                    print(
                        f"\n🧠 Agent 'Brain' glitch (Attempt {attempt+1}/{max_retries}). "
                        f"Retrying decision in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    raise e  # Real error, let it crash
        
        raise Exception("Agent Brain failed. Server is down.")


def create_agent(
    model_id: str = MODEL_ID,
    top_k_recipes: int = 1,
    max_steps: int = 12
) -> CulinaryAgent:
    """Factory function to create a culinary agent.
    
    Args:
        model_id: HuggingFace model ID for the LLM
        top_k_recipes: Number of recipes to retrieve per query
        max_steps: Maximum reasoning steps for the agent
        
    Returns:
        Initialized CulinaryAgent instance
    """
    return CulinaryAgent(
        model_id=model_id,
        top_k_recipes=top_k_recipes,
        max_steps=max_steps
    )
