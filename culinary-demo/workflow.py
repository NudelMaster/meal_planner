import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional, Union

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

# Find ~10 recipes via RAG, judge ranks them, UI surfaces the top 3 (rest held in reserve).
RETRIEVE_TOP_K = 15  # fetch a small buffer above 10 so dedup still leaves ~10 to judge
JUDGE_TOP_K = 10
MAX_EXCERPT_SECTION_CHARS = 250

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

class WebSearchEvent(Event):
    """Event to trigger web search."""
    query: str

# --- Prompts ---

WEB_SEARCH_PROMPT = PromptTemplate(
    template="""You are a culinary assistant. Extract and format recipes from the following web search results.
    
    User Query: "{user_query}"
    
    Web Results:
    {results_context}
    
    Instructions:
    - Extract distinct recipes found in the text.
    - Return a JSON array of objects with keys: "title", "recipe_text".
    - "recipe_text" should include Title, Ingredients, and Directions.
    - If ingredients/directions are missing, summarize what is available and include the Source URL.
    - Output JSON only.
    
    Formatted Recipes:"""
)

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

INTENT_ANALYZER_PROMPT = PromptTemplate(
    template="""Analyze this recipe request to understand exactly what the user wants.

    User Request: "{user_query}"

    Provide a structured analysis as JSON:
    {{
        "primary_goal": "what type of dish/meal",
        "requirements": [
            {{"attribute": "...", "qualifier": "high/low/with/without/etc", "look_for": "specific things to find"}}
        ],
        "restrictions": [
            {{"attribute": "...", "avoid": "specific things to avoid"}}
        ],
        "evaluation_focus": "describe what recipe sections matter and why"
    }}

    Rules:
    - Only include requirements and restrictions that the user EXPLICITLY stated, or that are inseparable from the request itself. Do NOT invent time limits, carb/calorie caps, allergen avoidances (nuts/gluten/dairy), or other dietary restrictions the user did not mention.
    - If the user stated no restrictions, return an empty "restrictions" list ([]). Do not add "avoid X if not specified" style guesses — a downstream judge treats every restriction as mandatory and will wrongly reject valid recipes.
    - Keep "requirements" to the core attribute(s) asked for (e.g., "high protein chicken" -> high protein AND chicken, nothing else).
    - The examples below map ONE explicit phrase to ONE entry; do not generalize beyond the words the user used.

    Examples:
    - "high protein" -> {"attribute": "protein", "qualifier": "high", "look_for": "meat, fish, eggs, legumes, tofu"}
    - "low carb" -> {"attribute": "carbs", "qualifier": "low", "avoid": "pasta, rice, bread, potatoes, sugar"}
    - "without nuts" -> {"attribute": "nuts", "avoid": "almonds, walnuts, peanuts, cashews, pine nuts"}
    - "quick meal" -> {"attribute": "time", "qualifier": "short", "look_for": "under 30 minutes"}
    - "very spicy" -> {"attribute": "spice", "qualifier": "high", "look_for": "hot peppers, chili, cayenne"}
    - "mild flavor" -> {"attribute": "spice", "qualifier": "low", "avoid": "hot peppers, strong spices"}

    Output JSON only.

    Analysis:"""
)

JUDGE_PROMPT = PromptTemplate(
    template="""You are an expert culinary judge. Evaluate recipes against the user's specific requirements.

    User Request: "{user_query}"

    Requirements (must have):
    {requirements}

    Restrictions (must avoid):
    {restrictions}

    Evaluation Focus:
    {evaluation_focus}

    Excluded Recipe Titles:
    {excluded_titles}

    Candidate Recipes (relevant content only):
    -------------------
    {candidates_str}
    -------------------

    Instructions:
    - A recipe must match the user's primary request (dish type, cuisine, or specific recipe name).
    - A recipe must satisfy all Requirements to be selected.
    - Reject any recipe that violates any Restrictions.
    - Exclude any recipe whose title matches the Excluded Recipe Titles.
    - IMPORTANT: If no recipes match the user's request, return an empty array: [].
    - It is better to return no results than to return irrelevant recipes.
    - Select up to {max_results} genuinely relevant recipes.
    - Ensure each recipe title is unique.
    - Return a JSON array with objects: {"title": "exact title", "reason": "brief explanation"}.
    - Rank results by relevance (best first).
    - Output JSON only.

    Selected Recipes:"""
)

FINAL_RESPONSE_PROMPT = PromptTemplate(
    template="""You are a helpful culinary assistant. Present the following recipes to the user in a clear, appetizing format.

    Recipes:
    {context_str}

    User Request: "{user_query}"

    Instructions:
    - Present ONLY the recipe exactly as given above. Use its EXACT ingredients and directions.
    - Do NOT add, remove, or substitute ingredients or steps. In particular, if the recipe has been adapted away from the original request, do NOT reintroduce ingredients the request mentions but the recipe no longer contains (e.g. keep a vegan adaptation vegan — do not add back chicken).
    - For each recipe, provide the Title (in bold), a one-sentence summary of the dish, the Ingredients list, and Directions.
    - Format it beautifully using Markdown.
    - Do NOT invent nutritional values.

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
    2. Retrieves a larger candidate set from Pinecone.
    3. Judges Relevance (Selects the most relevant set).
    4. Returns the selected payload.
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

    async def _increment_llm_calls(self, ctx: Context, label: str) -> None:
        """Track LLM calls per workflow run (logged to terminal only)."""
        try:
            count = await ctx.store.get("llm_call_count", 0)
        except Exception:
            count = 0
        count = int(count) + 1
        await ctx.store.set("llm_call_count", count)
        print(f"DEBUG: LLM call {count} - {label}")

    async def _llm_complete(
        self,
        ctx: Context,
        label: str,
        prompt: str,
        timeout: Optional[float] = None,
    ):
        """Run an LLM call with timing and prompt-size logging."""
        await self._increment_llm_calls(ctx, label)
        print(f"DEBUG: {label} prompt size: {len(prompt):,} chars")
        started = time.perf_counter()
        coro = self.llm.acomplete(prompt)
        if timeout:
            response = await asyncio.wait_for(coro, timeout=timeout)
        else:
            response = await coro
        elapsed = time.perf_counter() - started
        print(f"DEBUG: {label} finished in {elapsed:.1f}s")
        return response

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        text = text.strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    def _parse_json_object(self, text: str) -> Dict[str, Any]:
        """Parse a JSON object from LLM output."""
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return {}
            return json.loads(text[start : end + 1])
        except Exception:
            return {}

    def _parse_json_list(self, text: str) -> List[Dict[str, Any]]:
        """Parse a JSON array from LLM output."""
        try:
            start = text.find("[")
            end = text.rfind("]")
            if start == -1 or end == -1 or end <= start:
                return []
            payload = json.loads(text[start : end + 1])
            if isinstance(payload, list):
                return [item for item in payload if isinstance(item, dict)]
        except Exception:
            return []
        return []

    def _extract_title(self, recipe_text: str) -> str:
        """Extract a title from recipe text."""
        lines = [line.strip() for line in recipe_text.split("\n") if line.strip()]
        if not lines:
            return "Recipe"
        first = lines[0]
        if first.lower().startswith("title:"):
            return first.split(":", 1)[1].strip()
        for line in lines[:5]:
            lower = line.lower()
            if lower.startswith("recipe:") or lower.startswith("name:"):
                return line.split(":", 1)[1].strip()
        return first[:100]

    def _extract_section(self, recipe_text: str, headers: List[str]) -> str:
        """Extract a section by header keywords."""
        lower = recipe_text.lower()
        for header in headers:
            idx = lower.find(header)
            if idx != -1:
                remaining = recipe_text[idx:]
                # End at next header-like marker if present
                for next_header in ["ingredients", "directions", "instructions", "method", "steps", "prep", "cook", "time"]:
                    if next_header == header:
                        continue
                    next_idx = remaining.lower().find(next_header)
                    if next_idx > 0:
                        return remaining[:next_idx].strip()
                return remaining.strip()
        return ""

    def _extract_time_lines(self, recipe_text: str) -> str:
        """Extract lines mentioning time from recipe text."""
        lines = [line.strip() for line in recipe_text.split("\n") if line.strip()]
        time_lines = [
            line for line in lines
            if "time" in line.lower() or "min" in line.lower() or "hour" in line.lower()
        ]
        return "\n".join(time_lines[:5])

    def _build_candidate_excerpt(
        self,
        title: str,
        recipe_text: str,
        intent: Dict[str, Any],
        *,
        for_judge: bool = False,
    ) -> str:
        """Build a concise excerpt for judging based on intent."""
        evaluation_focus = str(intent.get("evaluation_focus", "")).lower()
        requirements = json.dumps(intent.get("requirements", []), ensure_ascii=True).lower()
        restrictions = json.dumps(intent.get("restrictions", []), ensure_ascii=True).lower()
        focus_text = " ".join([evaluation_focus, requirements, restrictions])

        include_ingredients = any(
            keyword in focus_text
            for keyword in [
                "ingredient", "protein", "carb", "calorie", "fat", "fiber", "sugar",
                "dairy", "gluten", "nut", "vegan", "vegetarian", "meat", "fish",
                "chicken", "beef", "tofu", "legume", "allergen", "spice",
            ]
        )

        include_directions = any(
            keyword in focus_text
            for keyword in [
                "direction", "instruction", "method", "step", "cook", "bake",
                "grill", "roast", "fry", "prep", "time", "quick", "slow",
            ]
        )

        include_time = any(keyword in focus_text for keyword in ["time", "minute", "hour", "quick", "slow"])

        parts: List[str] = [f"Title: {title}"]

        if include_ingredients:
            ingredients_section = self._extract_section(recipe_text, ["ingredients"]) or "Not specified"
            ingredients_section = self._truncate_text(ingredients_section, MAX_EXCERPT_SECTION_CHARS)
            parts.append(f"Ingredients:\n{ingredients_section}")

        if include_time:
            time_section = self._extract_time_lines(recipe_text) or "Not specified"
            time_section = self._truncate_text(time_section, 120)
            parts.append(f"Time:\n{time_section}")

        if include_directions and not for_judge:
            directions_section = (
                self._extract_section(recipe_text, ["directions", "instructions", "method", "steps"])
                or "Not specified"
            )
            directions_section = self._truncate_text(directions_section, MAX_EXCERPT_SECTION_CHARS)
            parts.append(f"Directions:\n{directions_section}")

        if len(parts) == 1:
            parts.append(self._truncate_text(recipe_text, MAX_EXCERPT_SECTION_CHARS))

        return "\n\n".join(parts)

    @step
    async def optimize_query(self, ctx: Context, ev: StartEvent) -> Union[OptimizeQueryEvent, WebSearchEvent]:
        """Translate user intent into a keyword-rich search query."""
        user_query = ev.get("query_str")
        search_mode = ev.get("search_mode", "db") # "db" or "web"
        excluded_titles = ev.get("excluded_titles", [])
        if not isinstance(excluded_titles, list):
            excluded_titles = [str(excluded_titles)] if excluded_titles else []

        if not user_query:
            return None

        await ctx.store.set("llm_call_count", 0)
        await ctx.store.set("user_query", user_query)
        await ctx.store.set("excluded_titles", excluded_titles)

        default_intent = {
            "primary_goal": user_query,
            "requirements": [],
            "restrictions": [],
            "evaluation_focus": "ingredients",
        }
        intent_prompt = INTENT_ANALYZER_PROMPT.format(user_query=user_query)

        if search_mode == "web":
            # Web search engines do better with the user's natural phrasing than with the
            # optimizer's keyword expansion (that expansion is tuned for vector/DB matching
            # and scatters web results). Skip the optimizer here — only the intent analysis
            # is needed downstream, to judge the web results.
            intent_response = await self._llm_complete(ctx, "intent_analyzer", intent_prompt)
            intent = self._parse_json_object(intent_response.text) or default_intent
            await ctx.store.set("intent_analysis", intent)
            print(f"DEBUG: Web search query (raw user request): '{user_query}'")
            return WebSearchEvent(query=user_query)

        # DB mode: intent analysis only needs the raw user query, so run it concurrently
        # with the query optimizer instead of after retrieval. This hides a full LLM
        # round-trip under the optimize+retrieve latency (eval_relevance reuses it from ctx).
        optimizer_prompt = QUERY_OPTIMIZER_PROMPT.format(user_query=user_query)
        optimizer_response, intent_response = await asyncio.gather(
            self._llm_complete(ctx, "query_optimizer", optimizer_prompt),
            self._llm_complete(ctx, "intent_analyzer", intent_prompt),
        )
        optimized_query = optimizer_response.text.strip()

        intent = self._parse_json_object(intent_response.text) or default_intent
        await ctx.store.set("intent_analysis", intent)

        print(f"DEBUG: Optimized Query: '{optimized_query}' (Mode: db)")
        return OptimizeQueryEvent(optimized_query=optimized_query)

    @step
    async def web_search(self, ctx: Context, ev: WebSearchEvent) -> StopEvent:
        """Search Tavily for recipes."""
        query = ev.query
        user_query = await ctx.store.get("user_query")
        
        print(f"DEBUG: Searching web for '{query}'...")
        try:
            # 1. Search Tavily
            search_result = self.tavily_client.search(
                query=f"{query} recipe full ingredients directions",
                search_depth="advanced",
                max_results=5
            )
            
            results_context = ""
            for i, res in enumerate(search_result.get("results", [])):
                results_context += f"Source {i+1}: {res['title']}\n{res['content']}\nURL: {res['url']}\n\n"

            if not results_context:
                return StopEvent(result="[]")

            # 2. Use LLM to format
            prompt = WEB_SEARCH_PROMPT.format(
                user_query=user_query,
                results_context=results_context
            )
            await self._increment_llm_calls(ctx, "web_search_formatter")
            response = await self.llm.acomplete(prompt)
            formatted_json = response.text.strip()
            web_recipes = self._parse_json_list(formatted_json)
            if not web_recipes:
                return StopEvent(result="[]")

            # Apply intent-aware filtering for web results if possible
            intent = await ctx.store.get("intent_analysis", None)
            if not isinstance(intent, dict):
                intent_prompt = INTENT_ANALYZER_PROMPT.format(user_query=user_query)
                await self._increment_llm_calls(ctx, "intent_analyzer_web")
                intent_response = await self.llm.acomplete(intent_prompt)
                intent = self._parse_json_object(intent_response.text)
                await ctx.store.set("intent_analysis", intent)

            candidates_str = ""
            recipe_map: Dict[str, Dict[str, Any]] = {}
            for idx, recipe in enumerate(web_recipes, 1):
                title = str(recipe.get("title") or f"Recipe {idx}")
                recipe_text = str(recipe.get("recipe_text") or "")
                recipe_map[title] = recipe
                excerpt = self._build_candidate_excerpt(title, recipe_text, intent)
                candidates_str += f"Recipe {idx}:\n{excerpt}\n\n"

            requirements_str = json.dumps(intent.get("requirements", []), indent=2) if isinstance(intent, dict) else "[]"
            restrictions_str = json.dumps(intent.get("restrictions", []), indent=2) if isinstance(intent, dict) else "[]"
            evaluation_focus = str(intent.get("evaluation_focus", "general match")) if isinstance(intent, dict) else "general match"
            excluded_titles = await ctx.store.get("excluded_titles", [])
            excluded_list = [str(title).strip() for title in excluded_titles if str(title).strip()]
            excluded_titles_str = "\n".join(f"- {title}" for title in excluded_list) or "None"

            judge_prompt = JUDGE_PROMPT.format(
                user_query=user_query,
                requirements=requirements_str,
                restrictions=restrictions_str,
                evaluation_focus=evaluation_focus,
                excluded_titles=excluded_titles_str,
                candidates_str=candidates_str,
                max_results=min(len(web_recipes), 10),
            )
            await self._increment_llm_calls(ctx, "judge_web_results")
            judge_response = await self.llm.acomplete(judge_prompt)
            selections = self._parse_json_list(judge_response.text)

            final_recipes: List[Dict[str, Any]] = []
            for selection in selections:
                selected_title = str(selection.get("title", "")).strip()
                if not selected_title:
                    continue
                if selected_title in recipe_map:
                    recipe = recipe_map[selected_title]
                    recipe["match_reason"] = selection.get("reason", "")
                    final_recipes.append(recipe)

            if not final_recipes:
                return StopEvent(result=json.dumps(web_recipes))

            return StopEvent(result=json.dumps(final_recipes))
            
        except Exception as e:
            print(f"ERROR: Web search failed: {e}")
            return StopEvent(result="[]")

    @step
    async def retrieve(self, ctx: Context, ev: OptimizeQueryEvent) -> RetrieveEvent:
        """Retrieve candidate nodes from Pinecone using the optimized query."""
        optimized_query = ev.optimized_query

        # Create retriever from the index
        # Fetch more to enable pagination in the UI
        retriever = self.index.as_retriever(similarity_top_k=RETRIEVE_TOP_K)
        nodes = retriever.retrieve(optimized_query)
        
        print(f"DEBUG: Retrieved {len(nodes)} nodes from Pinecone.")
        return RetrieveEvent(retrieved_nodes=nodes)

    @step
    async def eval_relevance(self, ctx: Context, ev: RetrieveEvent) -> QueryEvent:
        """Evaluate candidates and select the most relevant recipes."""
        user_query = await ctx.store.get("user_query")
        excluded_titles = await ctx.store.get("excluded_titles", [])
        if not isinstance(excluded_titles, list):
            excluded_titles = [str(excluded_titles)] if excluded_titles else []
        retrieved_nodes = ev.retrieved_nodes
        
        if not retrieved_nodes:
            return QueryEvent(context_str="No matching recipes found.")

        # Deduplicate by title and build map for full content (retrieval order = relevance)
        node_map: Dict[str, str] = {}
        candidates: List[Dict[str, str]] = []
        for node in retrieved_nodes:
            title = self._extract_title(node.text)
            if title in node_map:
                continue
            node_map[title] = node.text
            candidates.append({"title": title, "recipe_text": node.text})

        judge_candidates = candidates[:JUDGE_TOP_K]
        print(f"DEBUG: Judging top {len(judge_candidates)} of {len(candidates)} candidates.")
        print("DEBUG: Candidate titles: " + " | ".join(c["title"] for c in judge_candidates))

        # Intent was already computed concurrently in optimize_query; reuse it.
        intent = await ctx.store.get("intent_analysis", None)
        if not isinstance(intent, dict) or not intent:
            intent = {
                "primary_goal": user_query,
                "requirements": [],
                "restrictions": [],
                "evaluation_focus": "ingredients",
            }
        print(f"DEBUG: Intent primary_goal: {intent.get('primary_goal')!r}")
        print(f"DEBUG: Intent requirements: {json.dumps(intent.get('requirements', []))}")
        print(f"DEBUG: Intent restrictions: {json.dumps(intent.get('restrictions', []))}")

        candidates_str = ""
        for i, recipe in enumerate(judge_candidates, 1):
            excerpt = self._build_candidate_excerpt(
                recipe["title"], recipe["recipe_text"], intent, for_judge=True
            )
            candidates_str += f"Recipe {i}:\n{excerpt}\n\n"

        excluded_list = [str(title).strip() for title in excluded_titles if str(title).strip()]
        excluded_titles_str = "\n".join(f"- {title}" for title in excluded_list) or "None"
        max_results = min(len(judge_candidates), 10)
        requirements_str = json.dumps(intent.get("requirements", []), indent=2)
        restrictions_str = json.dumps(intent.get("restrictions", []), indent=2)
        evaluation_focus = str(intent.get("evaluation_focus", "general match"))

        prompt = JUDGE_PROMPT.format(
            user_query=user_query,
            requirements=requirements_str,
            restrictions=restrictions_str,
            evaluation_focus=evaluation_focus,
            excluded_titles=excluded_titles_str,
            candidates_str=candidates_str,
            max_results=max_results,
        )

        response = await self._llm_complete(ctx, "judge_candidates", prompt)
        raw_judge = response.text.strip()
        selections = self._parse_json_list(raw_judge)
        print(f"DEBUG: Judge returned {len(selections)} selection(s): "
              + " | ".join(str(s.get("title", "")) for s in selections))

        # Assemble full content for selected recipes
        final_recipes: List[Dict[str, Any]] = []
        unmatched: List[str] = []
        for selection in selections:
            selected_title = str(selection.get("title", "")).strip()
            if not selected_title:
                continue
            if selected_title in node_map:
                final_recipes.append({
                    "title": selected_title,
                    "recipe_text": node_map[selected_title],
                    "match_reason": selection.get("reason", "")
                })
            else:
                unmatched.append(selected_title)

        if unmatched:
            # Judge picked titles that don't exactly match the candidate titles we sent —
            # these are silently dropped. Distinguishes a title-mismatch from a true empty result.
            print(f"DEBUG: {len(unmatched)} judge title(s) did not match any candidate "
                  f"and were dropped: {' | '.join(unmatched)}")

        if not final_recipes:
            if not selections:
                print(f"DEBUG: Judge selected no recipes (returned empty/unparseable). "
                      f"Raw head: {raw_judge[:200]!r}")
            else:
                print("DEBUG: Judge selected recipes but none matched candidate titles (title mismatch).")
            return QueryEvent(context_str="[]")

        print("DEBUG: Judge selected recipes.")
        return QueryEvent(context_str=json.dumps(final_recipes))

    @step
    async def decide(self, ctx: Context, ev: QueryEvent) -> StopEvent:
        """Return the selected recipes payload."""
        return StopEvent(result=ev.context_str)
