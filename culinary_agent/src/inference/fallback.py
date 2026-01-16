"""Web search fallback tool using DuckDuckGo."""

from typing import List, Dict, Any
from smolagents import Tool

try:
    from duckduckgo_search import DDGS
except ImportError:
    # Fallback if specific version requires different import
    from ddgs import DDGS

class WebSearchTool(Tool):
    """Searches the web for recipes using DuckDuckGo."""
    
    name = "duckduckgo_search"
    # --- SMART DESCRIPTION FOR THE AGENT ---
    description = (
        "Searches the live web for recipes. "
        "USE THIS WHEN: "
        "1. The local database returns 0 results. "
        "2. The user has a very specific request (e.g. 'vegan lasagna without soy') "
        "that requires finding a totally new recipe."
    )
    inputs = {
        "query": {
            "type": "string", 
            "description": "Specific search query (e.g. 'vegan lasagna without soy recipe')."
        }
    }
    output_type = "string"

    def forward(self, query: str) -> str:
        """Search the web for recipe information."""
        try:
            # Use a context manager if possible, or direct instantiation
            results: List[Dict[str, Any]] = DDGS().text(query, max_results=3)
            
            if not results:
                return "No recipes found on the web."
            
            # Format the list of results
            formatted_results = f"Web Search Results for '{query}':\n"
            for res in results:
                title = res.get('title', 'Unknown Title')
                body = res.get('body', 'No content')
                url = res.get('href', '#')
                
                formatted_results += (
                    f"TITLE: {title}\n"
                    f"CONTENT: {body}\n"
                    f"URL: {url}\n\n"
                )
            
            return formatted_results
        except Exception as e:
            return f"Web search failed: {str(e)}"