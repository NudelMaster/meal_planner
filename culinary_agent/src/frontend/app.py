import os
import sys
import streamlit as st

# --- 1. PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up 2 levels to culinary_agent
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
# Point to the src folder directly
src_root = os.path.join(project_root, "src")

# Add both to path to catch all import styles
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_root not in sys.path:
    sys.path.insert(0, src_root)

# --- 2. LAZY INITIALIZATION ---
# We do NOT import the agent here. We import it inside the function.
# This prevents the app from crashing on startup if the agent has errors.

@st.cache_resource
def get_agent():
    """Imports and initializes the agent safely."""
    try:
        # --- MOVING IMPORTS INSIDE HERE ---
        # This allows Streamlit to start up even if imports are broken
        try:
            from agent.core import create_agent
        except ImportError:
            # Fallback for different folder structures
            from src.agent.core import create_agent
            
        # Initialize
        return create_agent(top_k_recipes=1, max_steps=15)
        
    except Exception as e:
        # This will print the REAL error to the UI instead of crashing
        st.error(f"🔥 AGENT CRASHED DURING LOADING: {e}")
        raise e

def load_session_state():
    try:
        # Import StateManager lazily too
        try:
            from utils.state_manager import StateManager
        except ImportError:
            from src.utils.state_manager import StateManager
            
        state_manager = StateManager()
        if 'state_manager' not in st.session_state:
            st.session_state.state_manager = state_manager
        
        if 'current_plan' not in st.session_state:
            if state_manager.has_state():
                st.session_state.current_plan = state_manager.load_state()
            else:
                st.session_state.current_plan = None
    except Exception as e:
        st.warning(f"State Manager failed (Non-Critical): {e}")

def main():
    st.set_page_config(page_title="Culinary Agent CLI-GUI", page_icon="🍳", layout="wide")
    
    st.title("🍳 Smart Culinary Agent")
    st.markdown("### Interactive Demo Mode")

    # Initialize State
    load_session_state()
    
    # Check if agent works
    agent = None
    try:
        with st.spinner("Waking up the agent..."):
            agent = get_agent()
            st.success("System Online")
    except Exception as e:
        st.error("🛑 System Halted. See error above.")
        st.stop()

    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.header("📋 Planning Controls")
        
        diet_request = st.text_input("Dietary Preferences", value="Balanced diet")
        
        has_plan = st.session_state.get('current_plan') is not None
        
        mode_options = [
            "1. Generate Full Day Plan",
            "2. Modify Breakfast Only",
            "3. Modify Lunch Only",
            "4. Modify Dinner Only"
        ]
        
        choice_idx = 0 if has_plan else 0
        selected_mode = st.radio("Select Action:", mode_options, index=choice_idx)
        
        col1, col2 = st.columns(2)
        with col1:
            run_btn = st.button("🚀 Execute", type="primary", use_container_width=True)
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                if hasattr(st.session_state, 'state_manager'):
                    st.session_state.state_manager.clear_state()
                st.session_state.current_plan = None
                st.rerun()

    # --- EXECUTION LOGIC ---
    if run_btn and agent:
        # 1. Validate Mode
        if not has_plan and "Full Day" not in selected_mode:
            st.warning("⚠️ Switching to Full Day Plan (No previous plan exists).")
            selected_mode = "1. Generate Full Day Plan"

        # 2. Construct Request
        if "1." in selected_mode:
            user_request = f"Create a full daily meal plan. Constraint: {diet_request}"
        elif "2." in selected_mode:
            user_request = f"Update ONLY the Breakfast. Constraint: {diet_request}"
        elif "3." in selected_mode:
            user_request = f"Update ONLY the Lunch. Constraint: {diet_request}"
        elif "4." in selected_mode:
            user_request = f"Update ONLY the Dinner. Constraint: {diet_request}"
        
        # 3. Construct Prompt
        # Need to import SYSTEM_PROMPT lazily too
        try:
            from agent.prompts import SYSTEM_PROMPT
        except ImportError:
            from src.agent.prompts import SYSTEM_PROMPT

        current_context = st.session_state.get('current_plan')
        if not current_context:
            full_prompt = f"{SYSTEM_PROMPT}\n\nUSER REQUEST: {user_request}"
        else:
            full_prompt = f"{SYSTEM_PROMPT}\n\nCONTEXT: {current_context}\n\nUSER REQUEST: {user_request}"

        # 4. Run
        with st.spinner(f"Agent is thinking..."):
            try:
                if hasattr(agent, 'run_with_retry'):
                     response = agent.run_with_retry(full_prompt)
                else:
                     response = agent.run(full_prompt)
                
                result_text = str(response)
                st.session_state.current_plan = result_text
                if hasattr(st.session_state, 'state_manager'):
                    st.session_state.state_manager.save_state(result_text)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Execution Error: {e}")

    # --- DISPLAY ---
    if st.session_state.get('current_plan'):
        st.subheader("🍽️ Current Meal Plan")
        st.markdown(st.session_state.current_plan)

if __name__ == "__main__":
    main()