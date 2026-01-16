import os
import sys
import streamlit as st

# --- 1. ROBUST PATH SETUP ---
# Forces the app to find the root folder properly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
src_root = os.path.join(project_root, "src")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_root not in sys.path:
    sys.path.insert(0, src_root)

# --- 2. SETUP UI FIRST (Passes Health Check) ---
st.set_page_config(page_title="Culinary Agent", page_icon="🍳", layout="wide")
st.title("🍳 Smart Culinary Agent")

# --- 3. THE "HEAVY" LOADER FUNCTION ---
@st.cache_resource(show_spinner=False)
def load_heavy_agent():
    """
    This function does the heavy lifting.
    We only call it when the user clicks the button.
    """
    # Import inside the function to prevent startup crashes
    try:
        try:
            from agent.core import create_agent
        except ImportError:
            from src.agent.core import create_agent
            
        # This is the slow part
        return create_agent(top_k_recipes=1, max_steps=15)
    except Exception as e:
        raise e

# --- 4. STATE MANAGEMENT ---
if 'agent_loaded' not in st.session_state:
    st.session_state.agent_loaded = False
if 'current_plan' not in st.session_state:
    st.session_state.current_plan = None

# --- 5. INITIALIZATION SCREEN (The Fix) ---
# If the agent isn't loaded yet, show a big button instead of crashing
if not st.session_state.agent_loaded:
    st.info("👋 Welcome! The system is ready to initialize.")
    st.markdown("Click the button below to load the AI models (FAISS + LLM).")
    
    if st.button("🔌 Initialize System", type="primary"):
        status_container = st.empty()
        try:
            with status_container.status("Initializing AI System...", expanded=True) as s:
                st.write("📂 Setting up paths...")
                # Verify imports work
                try:
                    from agent.prompts import SYSTEM_PROMPT
                    st.write("✅ Imports successful")
                except ImportError:
                    from src.agent.prompts import SYSTEM_PROMPT
                    st.write("✅ Imports successful (using fallback path)")
                
                st.write("🧠 Loading Agent (This may take 1-2 minutes)...")
                # This is where it usually crashes. Now we can see it happening.
                agent = load_heavy_agent()
                
                st.session_state.agent = agent
                st.session_state.agent_loaded = True
                s.update(label="System Online!", state="complete", expanded=False)
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ CRITICAL LOAD ERROR:\n\n{e}")
            st.stop()
    
    # STOP HERE until initialized
    st.stop()

# --- 6. MAIN APP (Only runs after button click) ---
# Getting the agent from session state is instant
agent = st.session_state.agent

# Initialize State Manager Lazily
try:
    try:
        from utils.state_manager import StateManager
    except ImportError:
        from src.utils.state_manager import StateManager
    state_manager = StateManager()
    if not st.session_state.current_plan and state_manager.has_state():
        st.session_state.current_plan = state_manager.load_state()
except:
    pass # Ignore state manager errors on cloud

# Sidebar & Main Logic
with st.sidebar:
    st.header("📋 Planning Controls")
    st.success("🟢 System Online")
    
    diet_request = st.text_input("Dietary Preferences", value="Balanced diet")
    
    mode_options = [
        "1. Generate Full Day Plan",
        "2. Modify Breakfast Only",
        "3. Modify Lunch Only",
        "4. Modify Dinner Only"
    ]
    selected_mode = st.radio("Select Action:", mode_options)
    
    col1, col2 = st.columns(2)
    with col1:
        run_btn = st.button("🚀 Execute", type="primary")
    with col2:
        if st.button("🗑️ Reset"):
            st.session_state.current_plan = None
            st.rerun()

if run_btn:
    # 1. Prepare Request
    has_plan = st.session_state.current_plan is not None
    if "1." in selected_mode or not has_plan:
        user_request = f"Create a full daily meal plan. Constraint: {diet_request}"
    elif "2." in selected_mode:
        user_request = f"Update ONLY the Breakfast. Constraint: {diet_request}"
    elif "3." in selected_mode:
        user_request = f"Update ONLY the Lunch. Constraint: {diet_request}"
    elif "4." in selected_mode:
        user_request = f"Update ONLY the Dinner. Constraint: {diet_request}"

    # 2. Prepare Prompt
    try:
        from agent.prompts import SYSTEM_PROMPT
    except:
        from src.agent.prompts import SYSTEM_PROMPT

    if st.session_state.current_plan:
        full_prompt = f"{SYSTEM_PROMPT}\n\nCONTEXT: {st.session_state.current_plan}\n\nUSER REQUEST: {user_request}"
    else:
        full_prompt = f"{SYSTEM_PROMPT}\n\nUSER REQUEST: {user_request}"

    # 3. Run
    with st.spinner("Agent is thinking..."):
        try:
            if hasattr(agent, 'run_with_retry'):
                response = agent.run_with_retry(full_prompt)
            else:
                response = agent.run(full_prompt)
            
            st.session_state.current_plan = str(response)
            st.rerun()
        except Exception as e:
            st.error(f"Execution Error: {e}")

if st.session_state.current_plan:
    st.subheader("🍽️ Current Meal Plan")
    st.markdown(st.session_state.current_plan)