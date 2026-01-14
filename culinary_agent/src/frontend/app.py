import os
import sys
import streamlit as st

# --- 1. SETUP PATHS (Monolith Mode) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- 2. IMPORTS FROM SRC ---
try:
    from src.core.agent import create_agent
    from src.prompts.templates import SYSTEM_PROMPT
    from src.utils.state_manager import StateManager
except ImportError as e:
    st.error(f"CRITICAL ERROR: Could not import project modules. Details: {e}")
    st.stop()

# --- 3. INITIALIZATION ---

@st.cache_resource
def get_agent():
    """Initialize the agent once (cached). Matches CLI parameters."""
    # Matches CLI: top_k=1, max_steps=15
    return create_agent(top_k_recipes=1, max_steps=15)

def load_session_state():
    """Syncs Streamlit session with the persistent StateManager on disk."""
    state_manager = StateManager()
    
    if 'state_manager' not in st.session_state:
        st.session_state.state_manager = state_manager
        
    # Check for previous session on disk if we don't have one in RAM
    if 'current_plan' not in st.session_state:
        if state_manager.has_state():
            st.session_state.current_plan = state_manager.load_state()
        else:
            st.session_state.current_plan = None

def main():
    st.set_page_config(page_title="Culinary Agent CLI-GUI", page_icon="🍳", layout="wide")
    
    # Initialize State
    load_session_state()
    agent = get_agent()
    
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
        
        # 2. MODE SELECTION (Matches CLI Options 1-4)
        has_plan = st.session_state.current_plan is not None
        
        mode_options = [
            "1. Generate Full Day Plan",
            "2. Modify Breakfast Only",
            "3. Modify Lunch Only",
            "4. Modify Dinner Only"
        ]
        
        # Disable update options if no plan exists
        if not has_plan:
            st.info("ℹ️ Generate a plan first to enable specific meal updates.")
            # Only allow option 1
            choice_idx = 0 
            disabled_modes = True
        else:
            choice_idx = 0
            disabled_modes = False

        selected_mode = st.radio(
            "Select Action:", 
            mode_options, 
            index=choice_idx,
            disabled=False # We handle valid logic inside the button
        )

        # 3. ACTIONS
        col1, col2 = st.columns(2)
        with col1:
            run_btn = st.button("🚀 Execute", type="primary", use_container_width=True)
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.state_manager.clear_state()
                st.session_state.current_plan = None
                st.rerun()

    # --- MAIN EXECUTION LOGIC (Mirrors run_cli) ---
    if run_btn:
        # 1. Validate Mode
        # If user tries to update a specific meal but has no plan, force Full Day
        if not has_plan and "Full Day" not in selected_mode:
            st.warning("⚠️ Cannot update a specific meal without an existing plan. Switching to Full Day Plan.")
            selected_mode = "1. Generate Full Day Plan"

        # 2. Construct 'user_request' (Matches CLI logic)
        if "1." in selected_mode:
            user_request = f"Create a full daily meal plan. Constraint: {diet_request}"
        elif "2." in selected_mode:
            user_request = f"Update ONLY the Breakfast. Constraint: {diet_request}"
        elif "3." in selected_mode:
            user_request = f"Update ONLY the Lunch. Constraint: {diet_request}"
        elif "4." in selected_mode:
            user_request = f"Update ONLY the Dinner. Constraint: {diet_request}"
        
        # 3. Construct 'full_prompt' (Matches CLI logic)
        current_context = st.session_state.current_plan
        
        if not current_context:
            # Fresh Start
            full_prompt = f"{SYSTEM_PROMPT}\n\nUSER REQUEST: {user_request}"
        else:
            # Context Aware Update
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
                # Direct string pass, just like CLI
                if hasattr(agent, 'run_with_retry'):
                     response = agent.run_with_retry(full_prompt)
                else:
                     response = agent.run(full_prompt)
                
                # 5. Save State
                result_text = str(response)
                st.session_state.current_plan = result_text
                st.session_state.state_manager.save_state(result_text)
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ CRITICAL ERROR: {e}")

    # --- DISPLAY RESULTS ---
    if st.session_state.current_plan:
        st.subheader("🍽️ Current Meal Plan")
        st.markdown(st.session_state.current_plan)
        
        with st.expander("📝 View Raw Text / Manual Edit"):
            # Allow manual override (which also saves to disk via StateManager)
            edited_text = st.text_area("Edit Plan", st.session_state.current_plan, height=300)
            if st.button("Save Manual Edits"):
                st.session_state.current_plan = edited_text
                st.session_state.state_manager.save_state(edited_text)
                st.success("Saved!")
                st.rerun()
    else:
        st.info("👋 Welcome! Use the sidebar to generate your first meal plan.")

if __name__ == "__main__":
    main()