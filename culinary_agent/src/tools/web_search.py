"""Web search tool using DuckDuckGo."""

from typing import List, Dict, Any

from smolagents import Tool

try:
    from duckduckgo_search import DDGS
except ImportError:
    # Fallback to ddgs if duckduckgo_search is not available
    from ddgs import DDGS


class WebSearchTool(Tool):
    """Searches the web for recipes using DuckDuckGo."""
    
    name = "duckduckgo_search"
    description = "Searches the web for recipes. Returns the content of the best result as a string."
    inputs = {
        "query": {"type": "string", "description": "Search query."}
    }
    output_type = "string"

    def forward(self, query: str) -> str:
        """Search the web for recipe information.
        
        Args:
            query: The search query
            
        Returns:
            Formatted string with search results
        """
        try:
            results: List[Dict[str, Any]] = DDGS().text(query, max_results=3)
            if not results:
                return "No recipes found on the web."
            
            # Format the list of results into a single string for the Agent
            formatted_results = "Web Search Results:\n"
            for res in results:
                formatted_results += (
                    f"TITLE: {res['title']}\n"
                    f"CONTENT: {res['body']}\n"
                    f"URL: {res['href']}\n\n"
                )
            
            return formatted_results
        except Exception as e:
            return f"Web search failed: {str(e)}"