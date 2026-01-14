# culinary_agent/frontend/app.py
import streamlit as st
import requests
import json

# Configuration
API_URL = "http://127.0.0.1:8000/generate-plan"

def main():
    st.set_page_config(page_title="Culinary Agent", page_icon="🍳", layout="wide")

    # --- Header ---
    st.title("🍳 Smart Culinary Agent")
    st.markdown("Powered by **Qwen-72B** • Agentic RAG • FastAPI Backend")

    # --- Session State Management ---
    # This acts like the "memory" for the frontend user
    if 'current_plan' not in st.session_state:
        st.session_state.current_plan = None
    if 'history' not in st.session_state:
        st.session_state.history = []

    # --- Sidebar: Controls ---
    with st.sidebar:
        st.header("🎮 Planning Controls")
        
        # 1. THE DIET (The Constraint)
        # This matches `diet_request = input(...)` from CLI
        diet_input = st.text_input(
            "Target Diet / Constraint", 
            value="High Protein",
            help="e.g. 'Vegan', 'Keto', 'No Mushrooms'"
        )

        # 2. THE CONTEXT (The Previous Plan)
        # We need to know IF we can edit. 
        # In CLI this was `if current_recipe_context:`
        has_plan = st.session_state.current_plan is not None

        # 3. THE MODE SELECTION (The Menu)
        # This matches the `print("1. Full Day... 2. Breakfast...")` from CLI
        if not has_plan:
            # If no plan exists, you can only generate a new one
            mode_options = ["Full Day Plan"]
            st.info("ℹ️ Generate a plan first to unlock editing options.")
        else:
            # If a plan exists, you unlock the "Update" options
            mode_options = [
                "Full Day Plan (Overwrite)", 
                "Update Breakfast Only", 
                "Update Lunch Only", 
                "Update Dinner Only"
            ]
        
        selected_mode = st.radio("Select Action:", mode_options)

        # 4. PREPARE THE DATA FOR API
        # We map the friendly text to the strict keys expected by api.py
        # This logic ensures the API knows EXACTLY what to do
        mode_map = {
            "Full Day Plan": "Full Day",
            "Full Day Plan (Overwrite)": "Full Day",
            "Update Breakfast Only": "Breakfast",
            "Update Lunch Only": "Lunch",
            "Update Dinner Only": "Dinner"
        }
        api_mode = mode_map[selected_mode]

        # 5. EXECUTE
        if st.button("🚀 Execute Agent", type="primary"):
            
            # Construct Payload (Strictly matching api.py MealPlanRequest)
            payload = {
                "diet_constraints": diet_input,
                "mode": api_mode,
                # CRITICAL: We must send the OLD plan so the agent can read it
                # The CLI did this by loading from file; here we load from RAM.
                "previous_plan": st.session_state.current_plan 
            }
            
            with st.spinner("Agent is working..."):
                try:
                    response = requests.post(API_URL, json=payload)
                    if response.status_code == 200:
                        new_plan = response.json()["result"]
                        st.session_state.current_plan = new_plan
                        st.rerun() # Refresh page to show new plan
                    else:
                        st.error(f"API Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection Failed: {e}")

    # --- Main Display (The "Out" Log) ---
    if st.session_state.current_plan:
        st.subheader("🍽️ Current Meal Plan")
        st.markdown(st.session_state.current_plan)
        
        # Feature: "Manual Override" (Like loading a file in CLI)
        with st.expander("📝 Manually Edit / Paste Old Plan"):
            edited_text = st.text_area("Context", st.session_state.current_plan, height=300)
            if st.button("Save Manual Edits"):
                st.session_state.current_plan = edited_text
                st.rerun()
if __name__ == "__main__":
    main()