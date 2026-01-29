import os
from typing import List, Optional, Any
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context,
)
from llama_index.core.schema import NodeWithScore
from llama_index.core.prompts import PromptTemplate
from llama_index.core.llms import LLM
from tavily import TavilyClient

# --- Events ---

class RetrieveEvent(Event):
    """Event containing retrieved nodes from the vector store."""
    retrieved_nodes: List[NodeWithScore]

class OptimizeQueryEvent(Event):
    """Event containing the optimized search query."""
    optimized_query: str

class QueryEvent(Event):
    """Event triggered when we have valid context to answer."""
    context_str: str

# --- Prompts ---

QUERY_OPTIMIZER_PROMPT = PromptTemplate(
    template="""You are a culinary search expert. Your goal is to translate a user's natural language request into a specific, keyword-rich search query that will match recipe titles, ingredients, and categories.

    User Request: "{user_query}"

    Instructions:
    - Expand abstract concepts into concrete ingredients (e.g., "high protein" -> "chicken beans tofu fish").
    - Expand dietary needs (e.g., "vegan" -> "plant-based no-dairy no-meat").
    - Expand situational requests (e.g., "cold day" -> "soup stew warm comfort").
    - Keep it concise and keyword-focused.
    - Output ONLY the optimized query string.

    Optimized Query:"""
)

JUDGE_PROMPT = PromptTemplate(
    template="""You are a strict culinary judge. Your task is to select the best 3 recipes that match the user's specific request.

    User Request: "{user_query}"

    Candidate Recipes:
    -------------------
    {candidates_str}
    -------------------

    Instructions:
    - Evaluate each recipe against the User Request.
    - Check for specific constraints (e.g., "no pepper", "high protein", "quick").
    - Select exactly 3 recipes that best fit.
    - If fewer than 3 are relevant, select the best ones available.
    - Return a JSON array of objects with keys: "title", "recipe_text".
    - Each "recipe_text" should include the full recipe content as provided.
    - Output JSON only.

    Selected Recipes:"""
)

FINAL_RESPONSE_PROMPT = PromptTemplate(
    template="""You are a helpful culinary assistant. Present the following recipes to the user in a clear, appetizing format.

    Recipes:
    {context_str}

    User Request: "{user_query}"

    Instructions:
    - For each recipe, provide the Title (in bold), a one-sentence summary of why it fits their request, the Ingredients list, and Directions.
    - Format it beautifully using Markdown.
    - Do NOT invent nutritional values.
    - If the user asked for something specific (e.g., "no pepper") and you selected a recipe, ensure you mention if it's a good fit or if they need to omit an ingredient.

    Response:"""
)

ADAPTATION_PROMPT = PromptTemplate(
    template="""You are a culinary assistant adapting a recipe to better meet a user's goal.

    User Request: "{user_query}"
    Adaptation Goal: "{adaptation_goal}"
    Selected Recipe:
    {recipe_text}

    Instructions:
    - Provide exactly 3 options.
    - Option 1: ingredient swaps for the goal.
    - Option 2: add-ons or modifications that increase the goal.
    - Option 3: a new recipe aligned with the goal and user request.
    - Return a JSON array of objects with keys: "title", "approach", "summary", "ingredients", "directions".
    - "ingredients" should be a list of strings.
    - "directions" should be a list of strings.
    - Output JSON only.

    Options:"""
)

# --- Workflow ---

class CorrectiveRAGWorkflow(Workflow):
    """
    Corrective RAG Workflow that:
    1. Optimizes Query (User -> Keywords).
    2. Retrieves Top-10 from Pinecone.
    3. Judges Relevance (Selects Top-3).
    4. Returns top-3 payload.
    """

    def __init__(
        self,
        index,
        llm: LLM,
        tavily_api_key: str,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.index = index
        self.llm = llm
        self.tavily_client = TavilyClient(api_key=tavily_api_key)

    @step
    async def optimize_query(self, ctx: Context, ev: StartEvent) -> OptimizeQueryEvent:
        """Translate user intent into a keyword-rich search query."""
        user_query = ev.get("query_str")
        if not user_query:
            return None
        
        ctx.store.set("user_query", user_query)
        
        prompt = QUERY_OPTIMIZER_PROMPT.format(user_query=user_query)
        response = await self.llm.acomplete(prompt)
        optimized_query = response.text.strip()
        
        print(f"DEBUG: Optimized Query: '{optimized_query}'")
        return OptimizeQueryEvent(optimized_query=optimized_query)

    @step
    async def retrieve(self, ctx: Context, ev: OptimizeQueryEvent) -> RetrieveEvent:
        """Retrieve top-10 nodes from Pinecone using the optimized query."""
        optimized_query = ev.optimized_query
        
        # Create retriever from the index
        # Fetch 10 to give the Judge enough candidates
        retriever = self.index.as_retriever(similarity_top_k=10)
        nodes = retriever.retrieve(optimized_query)
        
        print(f"DEBUG: Retrieved {len(nodes)} nodes from Pinecone.")
        return RetrieveEvent(retrieved_nodes=nodes)

    @step
    async def eval_relevance(self, ctx: Context, ev: RetrieveEvent) -> QueryEvent:
        """Evaluate candidates and select the best 3."""
        user_query = ctx.store.get("user_query")
        retrieved_nodes = ev.retrieved_nodes
        
        if not retrieved_nodes:
            return QueryEvent(context_str="No matching recipes found.")

        # Format candidates for the LLM to read
        candidates_str = ""
        for i, node in enumerate(retrieved_nodes, 1):
            candidates_str += f"Recipe {i}:\n{node.text[:1500]}\n\n" # Truncate slightly to save tokens if needed

        prompt = JUDGE_PROMPT.format(
            user_query=user_query,
            candidates_str=candidates_str
        )
        
        response = await self.llm.acomplete(prompt)
        selected_context = response.text.strip()
        
        print("DEBUG: Judge selected top recipes.")
        return QueryEvent(context_str=selected_context)

    @step
    async def decide(self, ctx: Context, ev: QueryEvent) -> StopEvent:
        """Return the selected recipes payload."""
        return StopEvent(result=ev.context_str)
