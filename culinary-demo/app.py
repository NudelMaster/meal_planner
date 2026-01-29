import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from tavily import TavilyClient
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.llms import LLM
from llama_index.core.workflow import StopEvent
from llama_index.vector_stores.pinecone import PineconeVectorStore
try:
    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
except ImportError:
    from llama_index.embeddings.google import GeminiEmbedding as GoogleGenAIEmbedding
from llama_index.llms.cerebras import Cerebras
from pinecone import Pinecone

from workflow import ADAPTATION_PROMPT, FINAL_RESPONSE_PROMPT, CorrectiveRAGWorkflow

# Load environment variables
load_dotenv()

# Constants
INDEX_NAME = "culinary-demo"

# --- Page Config ---
st.set_page_config(page_title="Culinary Assistant", layout="wide")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "workflow" not in st.session_state:
    st.session_state.workflow = None

if "top_recipes" not in st.session_state:
    st.session_state.top_recipes = []

if "last_query" not in st.session_state:
    st.session_state.last_query = None

if "selected_recipe_index" not in st.session_state:
    st.session_state.selected_recipe_index = None

if "selected_recipe" not in st.session_state:
    st.session_state.selected_recipe = None

if "selected_recipe_response" not in st.session_state:
    st.session_state.selected_recipe_response = None

if "adaptation_goal" not in st.session_state:
    st.session_state.adaptation_goal = ""

if "adaptation_options" not in st.session_state:
    st.session_state.adaptation_options = []

if "saved_sessions" not in st.session_state:
    st.session_state.saved_sessions = []

if "current_recipe" not in st.session_state:
    st.session_state.current_recipe = None

if "adaptation_history" not in st.session_state:
    st.session_state.adaptation_history = []

if "all_recipes" not in st.session_state:
    st.session_state.all_recipes = []

if "shown_recipe_titles" not in st.session_state:
    st.session_state.shown_recipe_titles = set()

if "database_exhausted" not in st.session_state:
    st.session_state.database_exhausted = False

if "viewing_history_mode" not in st.session_state:
    st.session_state.viewing_history_mode = False

# --- Helper Functions ---

def scroll_to_bottom():
    """Scroll to the bottom of the page using JavaScript."""
    components.html(
        """
        <script>
            window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'});
        </script>
        """,
        height=0,
        width=0,
    )

def save_current_session():
    """Save the current session state to history."""
    if not st.session_state.last_query and not st.session_state.messages:
        return

    # Create session snapshot
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    is_adapted = len(st.session_state.adaptation_history) > 0
    note = " (Adapted)" if is_adapted else ""
    
    session_data = {
        "id": datetime.now().isoformat(),
        "timestamp": timestamp,
        "query": st.session_state.last_query or "No query",
        "label": f"{timestamp} - {st.session_state.last_query or 'Untitled'}{note}",
        "messages": st.session_state.messages,
        "top_recipes": st.session_state.top_recipes,
        "all_recipes": st.session_state.all_recipes,
        "selected_recipe": st.session_state.selected_recipe,
        "selected_recipe_index": st.session_state.selected_recipe_index,
        "selected_recipe_response": st.session_state.selected_recipe_response,
        "current_recipe": st.session_state.current_recipe,
        "adaptation_history": st.session_state.adaptation_history,
        "adaptation_options": st.session_state.adaptation_options,
        "adaptation_goal": st.session_state.adaptation_goal
    }
    
    # Check if session with same ID exists (update it), else append
    # But since we generate ID on save, this creates a new entry every time "New Session" is clicked
    # which is safer to avoid overwriting history unintentionally.
    st.session_state.saved_sessions.append(session_data)

def load_session(session_data):
    """Load a session from history."""
    st.session_state.messages = session_data.get("messages", [])
    st.session_state.top_recipes = session_data.get("top_recipes", [])
    st.session_state.all_recipes = session_data.get("all_recipes", [])
    st.session_state.shown_recipe_titles = {r.get("title") for r in st.session_state.all_recipes}
    st.session_state.last_query = session_data.get("query")
    st.session_state.selected_recipe = session_data.get("selected_recipe")
    st.session_state.selected_recipe_index = session_data.get("selected_recipe_index")
    st.session_state.selected_recipe_response = session_data.get("selected_recipe_response")
    st.session_state.current_recipe = session_data.get("current_recipe")
    st.session_state.adaptation_history = session_data.get("adaptation_history", [])
    st.session_state.adaptation_options = session_data.get("adaptation_options", [])
    st.session_state.adaptation_goal = session_data.get("adaptation_goal", "")
    st.session_state.viewing_history_mode = True
    st.session_state.database_exhausted = False

def clear_current_session():
    """Reset the current session state."""
    st.session_state.messages = []
    st.session_state.top_recipes = []
    st.session_state.all_recipes = []
    st.session_state.shown_recipe_titles = set()
    st.session_state.last_query = None
    st.session_state.selected_recipe_index = None
    st.session_state.selected_recipe = None
    st.session_state.selected_recipe_response = None
    st.session_state.current_recipe = None
    st.session_state.adaptation_history = []
    st.session_state.adaptation_options = []
    st.session_state.adaptation_goal = ""
    st.session_state.viewing_history_mode = False
    st.session_state.database_exhausted = False

# --- Resource Loading ---
def initialize_resources():
    """Initialize DB connection and Models."""
    
    # Check API Keys
    required_keys = ["GOOGLE_API_KEY", "PINECONE_API_KEY", "CEREBRAS_API_KEY", "TAVILY_API_KEY"]
    missing = [k for k in required_keys if not os.getenv(k)]
    if missing:
        st.error(f"Missing API Keys: {', '.join(missing)}")
        st.stop()

    # 1. Initialize Embeddings (Google)
    try:
        from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
        embed_model = GoogleGenAIEmbedding(
            api_key=os.getenv("GOOGLE_API_KEY"),
            model_name="models/text-embedding-004",
        )
    except ImportError:
        from llama_index.embeddings.google import GeminiEmbedding as GoogleGenAIEmbedding
        embed_model = GoogleGenAIEmbedding(
            api_key=os.getenv("GOOGLE_API_KEY"),
            model_name="models/text-embedding-004",
        )
    
    Settings.embed_model = embed_model

    # 2. Initialize LLM (Cerebras Llama 3.1-70b)
    llm = Cerebras(
        model="llama-3.3-70b",
        api_key=os.getenv("CEREBRAS_API_KEY"),
        # Adjust temperature as needed
        temperature=0.3
    )
    Settings.llm = llm

    # 3. Connect to Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pinecone_index = pc.Index(INDEX_NAME)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    
    # 4. Create Index View (doesn't re-ingest, just connects)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )
    
    return index, llm

# --- Main App Logic ---

# Sidebar
with st.sidebar:
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/1830/1830839.png", width=50)
    with col2:
        st.title("Culinary Agent")
    
    st.caption(f"Pinecone: `{INDEX_NAME}` | Cerebras | Tavily")
    
    st.markdown("---")
    
    if st.button("➕ New Session", use_container_width=True):
        save_current_session()
        clear_current_session()
        st.rerun()
        
    st.markdown("### Session History")
    if not st.session_state.saved_sessions:
        st.info("No saved sessions yet.")
    else:
        # Show latest first
        for idx, session in enumerate(reversed(st.session_state.saved_sessions)):
            if st.button(f"📂 {session['label']}", key=f"hist_{idx}", use_container_width=True):
                # Save current work before switching
                if not st.session_state.viewing_history_mode:
                     save_current_session()
                
                load_session(session)
                st.rerun()

# Main Content
st.title("🍳 AI Culinary Assistant")
st.caption("Powered by Google Embeddings, Pinecone, Cerebras Llama 3.1, and Tavily")

# Custom CSS for Scrollable Container
st.markdown("""
<style>
    .recipe-container {
        max-height: 60vh;
        overflow-y: auto;
        padding-right: 10px;
        margin-bottom: 20px;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        background-color: #f9f9f9;
    }
    .recipe-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #eee;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .recipe-title {
        font-weight: bold;
        font-size: 1.1em;
        margin-bottom: 8px;
        color: #333;
    }
    .recipe-snippet {
        font-size: 0.9em;
        color: #666;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

try:
    index, llm = initialize_resources()
    workflow = CorrectiveRAGWorkflow(
        index=index,
        llm=llm,
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        verbose=True,
        timeout=120
    )
except Exception as e:
    st.error(f"Initialization Failed: {e}")
    st.stop()

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def extract_json_array(text: str) -> Optional[str]:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def parse_json_list(text: str) -> List[Dict[str, Any]]:
    payload = extract_json_array(text)
    if not payload:
        return []
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def normalize_recipes(items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for item in items:
        title = str(item.get("title") or item.get("name") or "Recipe")
        recipe_text = str(item.get("recipe_text") or item.get("text") or "")
        if recipe_text:
            normalized.append({"title": title, "recipe_text": recipe_text})
    return normalized


def as_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


async def run_workflow_async(workflow: CorrectiveRAGWorkflow, query: str, mode: str = "db") -> StopEvent:
    handler = workflow.run(query_str=query, search_mode=mode)
    return await handler


async def format_selected_recipe(llm: LLM, user_query: str, recipe_text: str) -> str:
    prompt = FINAL_RESPONSE_PROMPT.format(user_query=user_query, context_str=recipe_text)
    response = await llm.acomplete(prompt)
    return response.text.strip()


async def generate_adaptations(
    llm: LLM,
    user_query: str,
    recipe_text: str,
    adaptation_goal: str,
) -> str:
    prompt = ADAPTATION_PROMPT.format(
        user_query=user_query,
        recipe_text=recipe_text,
        adaptation_goal=adaptation_goal,
    )
    response = await llm.acomplete(prompt)
    return response.text.strip()

# Handle User Input
if prompt := st.chat_input("Ask for a recipe or cooking tip..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                # If we are already viewing a recipe or history, a new prompt should reset the search
                # But keep the session unless "New Session" is clicked.
                # Just clear specific recipe state for new search
                st.session_state.top_recipes = []
                st.session_state.all_recipes = []
                st.session_state.shown_recipe_titles = set()
                st.session_state.selected_recipe = None
                st.session_state.current_recipe = None
                st.session_state.selected_recipe_index = None
                st.session_state.selected_recipe_response = None
                st.session_state.adaptation_history = []
                st.session_state.adaptation_options = []
                st.session_state.database_exhausted = False
                
                result = asyncio.run(run_workflow_async(workflow, prompt))
                raw_result = result.result if isinstance(result, StopEvent) else str(result)
                recipes = normalize_recipes(parse_json_list(raw_result))
                if not recipes:
                    st.error("No recipes were returned. Please try another query.")
                else:
                    st.session_state.top_recipes = recipes
                    st.session_state.all_recipes.extend(recipes)
                    for r in recipes:
                        st.session_state.shown_recipe_titles.add(r.get("title"))
                    
                    st.session_state.last_query = prompt
                    st.session_state.viewing_history_mode = False # Edited new content -> active mode
                    
                    summary = "Found recipes. Choose one below."
                    st.markdown(summary)
                    st.session_state.messages.append({"role": "assistant", "content": summary})
                    scroll_to_bottom()
        except Exception as e:
            import traceback
            st.error(f"Error occurred: {e}")
            st.error(f"Traceback:\n{traceback.format_exc()}")

if st.session_state.all_recipes:
    st.subheader(f"Found Recipes ({len(st.session_state.all_recipes)})")
    
    # Scrollable Container
    with st.container(height=500):
        for idx, recipe in enumerate(st.session_state.all_recipes):
            title = recipe.get("title", "Recipe")
            recipe_text = recipe.get("recipe_text", "")
            snippet = recipe_text[:300] + "..." if len(recipe_text) > 300 else recipe_text
            is_web = recipe.get("source") == "web"
            
            # Recipe Card
            with st.container():
                st.markdown(f"**Option {idx + 1}: {title}** {'🌐' if is_web else ''}")
                st.caption(snippet)
                if recipe_text:
                    with st.expander("View full recipe"):
                        st.markdown(recipe_text)
                if st.button(f"Select Option {idx + 1}", key=f"select_recipe_{idx}", use_container_width=True):
                    st.session_state.selected_recipe_index = idx
                    st.session_state.selected_recipe = recipe
                    st.session_state.current_recipe = recipe # Set as current for adaptation
                    st.session_state.selected_recipe_response = None
                    st.session_state.adaptation_options = []
                    st.session_state.adaptation_goal = ""
                    st.session_state.adaptation_history = []
                    # Don't scroll to bottom here, just rerun to show selected recipe below
    
    # Action Buttons (Generate More / Search Online)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🔄 Generate More Recipes", use_container_width=True, disabled=st.session_state.database_exhausted):
            with st.spinner("Searching for more unique recipes..."):
                # Run workflow again
                try:
                    # We might need to handle this in a separate function to avoid code duplication
                    # But for now, inline is fine
                    new_result = asyncio.run(run_workflow_async(workflow, st.session_state.last_query))
                    raw_new = new_result.result if isinstance(new_result, StopEvent) else str(new_result)
                    new_recipes = normalize_recipes(parse_json_list(raw_new))
                    
                    # Filter duplicates
                    unique_new = []
                    for r in new_recipes:
                        if r.get("title") not in st.session_state.shown_recipe_titles:
                            unique_new.append(r)
                            st.session_state.shown_recipe_titles.add(r.get("title"))
                    
                    if unique_new:
                        st.session_state.all_recipes.extend(unique_new)
                        st.success(f"Found {len(unique_new)} new recipes!")
                        st.rerun()
                    else:
                        st.session_state.database_exhausted = True
                        st.warning("No more unique recipes found in database.")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error fetching more recipes: {e}")

    with col2:
        if st.session_state.database_exhausted:
            if st.button("🌐 Search Online (Tavily)", use_container_width=True):
                with st.spinner("Searching the web..."):
                    # Use Workflow with search_mode="web"
                    result = asyncio.run(run_workflow_async(workflow, st.session_state.last_query, mode="web"))
                    raw_result = result.result if isinstance(result, StopEvent) else str(result)
                    web_recipes = normalize_recipes(parse_json_list(raw_result))
                    
                    if web_recipes:
                        # Mark as web source
                        for r in web_recipes:
                            r["source"] = "web"
                            st.session_state.shown_recipe_titles.add(r.get("title"))
                        
                        st.session_state.all_recipes.extend(web_recipes)
                        st.success(f"Found {len(web_recipes)} recipes from the web!")
                        st.rerun()
                    else:
                        st.error("Could not find relevant recipes online.")

if st.session_state.current_recipe:
    st.markdown("---")
    st.subheader("Selected Recipe")
    
    # Show active recipe title
    active_title = st.session_state.current_recipe.get("title", "Recipe")
    active_text = st.session_state.current_recipe.get("recipe_text", "")
    
    st.markdown(f"### {active_title}")
    
    # Determine if we should show formatted or raw
    # Logic: If formatting was requested for THIS recipe, show it.
    # But since adaptation changes current_recipe, we lose formatting unless we re-format.
    # For now, keep the manual format button logic.
    
    # Show raw text in expander to keep UI clean
    with st.expander("View Raw Recipe Text", expanded=True):
        st.markdown(active_text)

    # Formatting Section
    if st.button("✨ Format this recipe", key="fmt_btn"):
        with st.spinner("Formatting recipe..."):
            user_query = st.session_state.last_query or ""
            try:
                formatted = asyncio.run(
                    asyncio.wait_for(
                        format_selected_recipe(llm, user_query, active_text),
                        timeout=90,
                    )
                )
                st.session_state.selected_recipe_response = formatted
                scroll_to_bottom()
            except asyncio.TimeoutError:
                st.error("Formatting timed out. Please try again.")
    
    if st.session_state.selected_recipe_response:
        st.markdown(st.session_state.selected_recipe_response)

    # Adaptation Section
    st.markdown("---")
    st.subheader("Adapt Recipe")
    
    # Show history of adaptations
    if st.session_state.adaptation_history:
        st.info(f"Recipe has been adapted {len(st.session_state.adaptation_history)} times.")
        with st.expander("View Adaptation History"):
            for i, hist in enumerate(st.session_state.adaptation_history):
                st.markdown(f"**Step {i+1}:** {hist['goal']} (from {hist['source_recipe']})")

    adaptation_goal = st.text_input(
        "How would you like to adapt this recipe?",
        value=st.session_state.adaptation_goal,
        key="adaptation_goal_input",
        placeholder="e.g., make it vegetarian, add more spice, swap chicken for tofu"
    )
    
    if st.button("Generate Adaptations", key="gen_adapt_btn"):
        if not adaptation_goal.strip():
            st.warning("Please enter an adaptation goal.")
        else:
            st.session_state.adaptation_goal = adaptation_goal.strip()
            # Clear previous options to avoid confusion
            st.session_state.adaptation_options = [] 
            
            with st.spinner("Generating adaptations..."):
                user_query = st.session_state.last_query or ""
                # ALWAYS adapt the CURRENT recipe (whether original or already adapted)
                recipe_text_to_adapt = st.session_state.current_recipe.get("recipe_text", "")
                
                raw = asyncio.run(
                    generate_adaptations(
                        llm,
                        user_query,
                        recipe_text_to_adapt,
                        st.session_state.adaptation_goal,
                    )
                )
                st.session_state.adaptation_options = parse_json_list(raw)
                scroll_to_bottom()

# Adaptation Options Display
if st.session_state.adaptation_options:
    st.markdown("### Adaptation Options")
    st.caption("Select an option to update the current recipe and continue adapting.")
    
    cols = st.columns(3)
    for idx, option in enumerate(st.session_state.adaptation_options):
        # Use columns for grid layout if possible, or just list
        title = str(option.get("title") or f"Option {idx + 1}")
        approach = str(option.get("approach") or "")
        summary = str(option.get("summary") or "")
        ingredients = as_string_list(option.get("ingredients"))
        directions = as_string_list(option.get("directions"))
        
        # Build new recipe object from this option
        # We need to construct a 'recipe_text' from the components so it can be adapted again
        new_recipe_text = f"Title: {title}\n\nSummary: {summary}\n\nIngredients:\n" + \
                          "\n".join([f"- {i}" for i in ingredients]) + \
                          "\n\nDirections:\n" + \
                          "\n".join([f"{i}" for i in directions])
        
        new_recipe_obj = {
            "title": title,
            "recipe_text": new_recipe_text,
            "ingredients": ingredients,
            "directions": directions
        }

        st.markdown(f"#### Option {idx+1}: {title}")
        if approach: st.markdown(f"*{approach}*")
        if summary: st.markdown(summary)
        
        with st.expander("Details"):
            if ingredients:
                st.markdown("**Ingredients:**")
                for item in ingredients: st.markdown(f"- {item}")
            if directions:
                st.markdown("**Directions:**")
                for item in directions: st.markdown(f"- {item}")
        
        if st.button(f"Use this version", key=f"use_adapt_{idx}"):
            # 1. Archive current state to history
            st.session_state.adaptation_history.append({
                "goal": st.session_state.adaptation_goal,
                "source_recipe": st.session_state.current_recipe.get("title", "Unknown"),
                "options_generated": len(st.session_state.adaptation_options)
            })
            
            # 2. Update current recipe to this new one
            st.session_state.current_recipe = new_recipe_obj
            
            # 3. Reset UI states for next round
            st.session_state.adaptation_options = []
            st.session_state.adaptation_goal = ""
            st.session_state.selected_recipe_response = None # Clear old formatting
            
            st.success(f"Updated recipe to: {title}")
            scroll_to_bottom()
            st.rerun()
            
    st.markdown("---")
