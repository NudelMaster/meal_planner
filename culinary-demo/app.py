import asyncio
import json
import os
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
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
    st.image("https://cdn-icons-png.flaticon.com/512/1830/1830839.png", width=100)
    st.title("Culinary Agent")
    st.markdown("---")
    st.success(f"Connected to Pinecone Index:\n`{INDEX_NAME}`")
    st.info("Workflow: Retrieve -> Judge (Cerebras) -> Decide -> Tavily (if needed)")

# Main Content
st.title("🍳 AI Culinary Assistant")
st.caption("Powered by Google Embeddings, Pinecone, Cerebras Llama 3.1, and Tavily")

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


async def run_workflow_async(workflow: CorrectiveRAGWorkflow, query: str) -> StopEvent:
    handler = workflow.run(query_str=query)
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
                result = asyncio.run(run_workflow_async(workflow, prompt))
                raw_result = result.result if isinstance(result, StopEvent) else str(result)
                recipes = normalize_recipes(parse_json_list(raw_result))
                if not recipes:
                    st.error("No recipes were returned. Please try another query.")
                else:
                    st.session_state.top_recipes = recipes
                    st.session_state.last_query = prompt
                    st.session_state.selected_recipe_index = None
                    st.session_state.selected_recipe = None
                    st.session_state.selected_recipe_response = None
                    st.session_state.adaptation_goal = ""
                    st.session_state.adaptation_options = []
                    summary = "Top 3 recipes ready. Choose one below."
                    st.markdown(summary)
                    st.session_state.messages.append({"role": "assistant", "content": summary})
        except Exception as e:
            import traceback
            st.error(f"Error occurred: {e}")
            st.error(f"Traceback:\n{traceback.format_exc()}")

if st.session_state.top_recipes:
    st.subheader("Top 3 recipes")
    for idx, recipe in enumerate(st.session_state.top_recipes):
        title = recipe.get("title", "Recipe")
        recipe_text = recipe.get("recipe_text", "")
        snippet = recipe_text[:400] + "..." if len(recipe_text) > 400 else recipe_text
        st.markdown(f"**Option {idx + 1}: {title}**")
        if snippet:
            st.markdown(snippet)
        if st.button(f"Select Option {idx + 1}", key=f"select_recipe_{idx}"):
            st.session_state.selected_recipe_index = idx
            st.session_state.selected_recipe = recipe
            st.session_state.selected_recipe_response = None
            st.session_state.adaptation_options = []
            st.session_state.adaptation_goal = ""
            st.session_state.selected_recipe_response = None

if st.session_state.selected_recipe:
    st.subheader("Selected recipe")
    selected_title = st.session_state.selected_recipe.get("title", "Recipe")
    selected_text = st.session_state.selected_recipe.get("recipe_text", "")
    st.markdown(f"**{selected_title}**")
    if selected_text:
        st.markdown(selected_text)
    if st.button("Format selected recipe"):
        with st.spinner("Formatting recipe..."):
            user_query = st.session_state.last_query or ""
            try:
                formatted = asyncio.run(
                    asyncio.wait_for(
                        format_selected_recipe(llm, user_query, selected_text),
                        timeout=90,
                    )
                )
                st.session_state.selected_recipe_response = formatted
            except asyncio.TimeoutError:
                st.error("Formatting timed out. Please try again.")
    if st.session_state.selected_recipe_response:
        st.markdown(st.session_state.selected_recipe_response)

if st.session_state.selected_recipe:
    st.subheader("Adapt recipe")
    adaptation_goal = st.text_input(
        "Adaptation goal",
        value=st.session_state.adaptation_goal,
        key="adaptation_goal_input",
    )
    if st.button("Generate adaptations"):
        if not adaptation_goal.strip():
            st.warning("Please enter an adaptation goal.")
        else:
            st.session_state.adaptation_goal = adaptation_goal.strip()
            st.session_state.adaptation_options = []
            with st.spinner("Generating adaptations..."):
                user_query = st.session_state.last_query or ""
                recipe_text = st.session_state.selected_recipe.get("recipe_text", "")
                raw = asyncio.run(
                    generate_adaptations(
                        llm,
                        user_query,
                        recipe_text,
                        st.session_state.adaptation_goal,
                    )
                )
                st.session_state.adaptation_options = parse_json_list(raw)

if st.session_state.adaptation_options:
    st.subheader("Adaptation options")
    for idx, option in enumerate(st.session_state.adaptation_options):
        title = str(option.get("title") or f"Option {idx + 1}")
        approach = str(option.get("approach") or "")
        summary = str(option.get("summary") or "")
        ingredients = as_string_list(option.get("ingredients"))
        directions = as_string_list(option.get("directions"))
        st.markdown(f"**{title}**")
        if approach:
            st.markdown(approach)
        if summary:
            st.markdown(summary)
        if ingredients:
            st.markdown("Ingredients:\n" + "\n".join([f"- {item}" for item in ingredients]))
        if directions:
            st.markdown("Directions:\n" + "\n".join([f"- {item}" for item in directions]))
