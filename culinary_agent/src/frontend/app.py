import os
import sys
import streamlit as st

# --- 1. PATH SETUP (Based on your Screenshot) ---
# Current file: .../culinary_agent/src/frontend/app.py
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go up 3 levels: frontend -> src -> culinary_agent
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))

# Add to Python Path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 2. DEBUGGING (Delete this section after it works) ---
# This will prove if we found the right folder
try:
    files_in_root = os.listdir(project_root)
    if "src" not in files_in_root:
        st.error(f"⚠️ ROOT ERROR: Expected to find 'src' folder in {project_root} but found: {files_in_root}")
        st.stop()
except Exception as e:
    st.error(f"Path Error: {e}")
# ---------------------------------------------------------

# --- 3. IMPORTS ---
try:
    from src.agent.core import CulinaryAgent, create_agent
    from src.agent.prompts import SYSTEM_PROMPT
    from src.utils.state_manager import StateManager
except ImportError as e:
    st.error(f"CRITICAL ERROR: {e}")
    st.write("Ensure your __init__.py files exist in src/ and src/agent/")
    st.stop()

# --- 3. INITIALIZATION ---

@st.cache_resource
def get_agent():
    """Initialize the agent once (cached)."""
    # Using the factory function from your structure
    return create_agent(top_k_recipes=1, max_steps=15)

def load_session_state():
    """Syncs Streamlit session with the persistent StateManager on disk."""
    try:
        state_manager = StateManager()
        
        if 'state_manager' not in st.session_state:
            st.session_state.state_manager = state_manager
            
        if 'current_plan' not in st.session_state:
            if state_manager.has_state():
                st.session_state.current_plan = state_manager.load_state()
            else:
                st.session_state.current_plan = None
    except Exception as e:
        # Fallback if state manager fails (e.g. permission issues on cloud)
        st.warning(f"State Manager warning: {e}. Session will be temporary.")
        if 'current_plan' not in st.session_state:
             st.session_state.current_plan = None

def main():
    st.set_page_config(page_title="Culinary Agent CLI-GUI", page_icon="🍳", layout="wide")
    
    # Initialize State
    load_session_state()
    
    try:
        agent = get_agent()
    except Exception as e:
        st.error(f"Agent Initialization Failed: {e}")
        st.stop()
    
    # --- UI HEADER ---
    st.title("🍳 Smart Culinary Agent")
    st.markdown("### Interactive Demo Mode")
    
    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.header("📋 Planning Controls")
        
        # 1. DIET INPUT
        diet_request = st.text_input(
            "Dietary Preferences", 
            value="Balanced diet",
            help="e.g. 'Vegan', 'High Protein', 'No Mushrooms'"
        )
        
        # 2. MODE SELECTION
        has_plan = st.session_state.current_plan is not None
        
        mode_options = [
            "1. Generate Full Day Plan",
            "2. Modify Breakfast Only",
            "3. Modify Lunch Only",
            "4. Modify Dinner Only"
        ]
        
        if not has_plan:
            st.info("ℹ️ Generate a plan first to enable specific meal updates.")
            choice_idx = 0 
        else:
            choice_idx = 0

        selected_mode = st.radio(
            "Select Action:", 
            mode_options, 
            index=choice_idx
        )

        # 3. ACTIONS
        col1, col2 = st.columns(2)
        with col1:
            run_btn = st.button("🚀 Execute", type="primary", use_container_width=True)
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                if hasattr(st.session_state, 'state_manager'):
                    st.session_state.state_manager.clear_state()
                st.session_state.current_plan = None
                st.rerun()

    # --- MAIN EXECUTION LOGIC ---
    if run_btn:
        # 1. Validate Mode
        if not has_plan and "Full Day" not in selected_mode:
            st.warning("⚠️ Cannot update a specific meal without an existing plan. Switching to Full Day Plan.")
            selected_mode = "1. Generate Full Day Plan"

        # 2. Construct 'user_request'
        if "1." in selected_mode:
            user_request = f"Create a full daily meal plan. Constraint: {diet_request}"
        elif "2." in selected_mode:
            user_request = f"Update ONLY the Breakfast. Constraint: {diet_request}"
        elif "3." in selected_mode:
            user_request = f"Update ONLY the Lunch. Constraint: {diet_request}"
        elif "4." in selected_mode:
            user_request = f"Update ONLY the Dinner. Constraint: {diet_request}"
        
        # 3. Construct 'full_prompt'
        current_context = st.session_state.current_plan
        
        if not current_context:
            full_prompt = f"{SYSTEM_PROMPT}\n\nUSER REQUEST: {user_request}"
        else:
            full_prompt = f"""
            {SYSTEM_PROMPT}

            CONTEXT: The user has an existing plan.
            PREVIOUS PLAN:
            {current_context}

            USER REQUEST: {user_request}

            TASK: 
            1. If the user asked for a FULL PLAN, ignore the previous plan and generate new.
            2. If the user asked to UPDATE a specific meal, keep the other meals from PREVIOUS PLAN and only rewrite the requested one.
            """

        # 4. Run Agent
        with st.spinner(f"Agent is thinking... [Task: {user_request}]"):
            try:
                # Check for run vs run_with_retry
                if hasattr(agent, 'run_with_retry'):
                     response = agent.run_with_retry(full_prompt)
                else:
                     response = agent.run(full_prompt)
                
                # 5. Save State
                result_text = str(response)
                st.session_state.current_plan = result_text
                
                if hasattr(st.session_state, 'state_manager'):
                    st.session_state.state_manager.save_state(result_text)
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ CRITICAL ERROR: {e}")

    # --- DISPLAY RESULTS ---
    if st.session_state.current_plan:
        st.subheader("🍽️ Current Meal Plan")
        st.markdown(st.session_state.current_plan)
        
        with st.expander("📝 View Raw Text / Manual Edit"):
            edited_text = st.text_area("Edit Plan", st.session_state.current_plan, height=300)
            if st.button("Save Manual Edits"):
                st.session_state.current_plan = edited_text
                if hasattr(st.session_state, 'state_manager'):
                    st.session_state.state_manager.save_state(edited_text)
                st.success("Saved!")
                st.rerun()
    else:
        st.info("👋 Welcome! Use the sidebar to generate your first meal plan.")

if __name__ == "__main__":
    main()
