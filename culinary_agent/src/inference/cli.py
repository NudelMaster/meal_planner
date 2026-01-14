"""CLI entry point for the culinary agent."""

import sys
from pathlib import Path

from src.core.agent import create_agent
from src.prompts.templates import SYSTEM_PROMPT
from src.utils.state_manager import StateManager


def run_cli():
    """Run the CLI interface for the culinary agent."""
    print("="*60)
    print(" 🍳 CULINARY AGENT - Smart Daily Meal Planner")
    print("="*60)
    
    # Initialize state manager
    state_manager = StateManager()
    
    # Initialize agent
    print("\nInitializing agent...")
    try:
        # Increased max_steps to allow for 3-meal iterations
        agent = create_agent(top_k_recipes=1, max_steps=15)
    except Exception as e:
        print(f"\n❌ Failed to initialize agent: {e}")
        print("\nPlease ensure:")
        print("1. Recipe data files exist in the project root")
        print("2. FAISS index has been built")
        print("3. Required dependencies are installed")
        sys.exit(1)
    
    # Check for previous session
    current_recipe_context = None
    if state_manager.has_state():
        print("\n📂 Found previous session! Loading last plan...")
        current_recipe_context = state_manager.load_state()
        if current_recipe_context:
            print(f"RESUMED CONTEXT (First 100 chars): {current_recipe_context[:100]}...")
    
    # Main interaction loop
    while True:
        print("\n" + "-"*60)
        print(" 📋 PLANNING MODE SELECTION:")
        print(" 1. Generate Full Day Plan (Breakfast, Lunch, Dinner)")
        print(" 2. Modify/Generate ONLY Breakfast")
        print(" 3. Modify/Generate ONLY Lunch")
        print(" 4. Modify/Generate ONLY Dinner")
        print(" c. Clear Session (Start Fresh)")
        print(" q. Quit")
        print("-"*60)

        choice = input("Select an option (1-4, c, q): ").strip().lower()

        # Handle Exits and Clears
        if choice in ['q', 'quit', 'exit']:
            print("\n👋 Goodbye! Your session has been saved.")
            break
        
        elif choice == 'c':
            current_recipe_context = None
            state_manager.clear_state()
            print("✓ Session cleared. Starting fresh.")
            continue

        # Get Dietary Constraints
        if choice in ['1', '2', '3', '4']:
            diet_request = input("Enter dietary preferences (e.g., 'vegan', 'high protein'): ").strip()
            if not diet_request:
                diet_request = "balanced diet" # Default fallback
        else:
            print("⚠️  Invalid choice, please try again.")
            continue

        # Construct the User Request based on Menu Choice
        # This string triggers the "Smart Targeting" logic in your System Prompt
        if choice == '1':
            user_request = f"Create a full daily meal plan. Constraint: {diet_request}"
        elif choice == '2':
            user_request = f"Update ONLY the Breakfast. Constraint: {diet_request}"
        elif choice == '3':
            user_request = f"Update ONLY the Lunch. Constraint: {diet_request}"
        elif choice == '4':
            user_request = f"Update ONLY the Dinner. Constraint: {diet_request}"

        # Construct the Final Prompt for the Agent
        # If we have context, we attach it so the agent knows what to "Update"
        if not current_recipe_context:
            full_prompt = f"{SYSTEM_PROMPT}\n\nUSER REQUEST: {user_request}"
        else:
            full_prompt = f"""
                            {SYSTEM_PROMPT}

                            CONTEXT: The user has an existing plan.
                            PREVIOUS PLAN:
                            {current_recipe_context}

                            USER REQUEST: {user_request}

                            TASK: 
                            1. If the user asked for a FULL PLAN, ignore the previous plan and generate new.
                            2. If the user asked to UPDATE a specific meal, keep the other meals from PREVIOUS PLAN and only rewrite the requested one.
                            """
        
        print("\n" + "="*60)
        print(f"🤔 Agent is thinking... [Task: {user_request}]")
        print("="*60 + "\n")
        
        try:
            # Run the agent
            # Note: Ensure agent.run_with_retry() exists in your core.py, or use agent.run()
            if hasattr(agent, 'run_with_retry'):
                response = agent.run_with_retry(full_prompt)
            else:
                response = agent.run(full_prompt)
            
            # Save state immediately
            current_recipe_context = str(response)
            state_manager.save_state(current_recipe_context)
            
            # Display response
            print("\n" + "="*60)
            print("🍽️  AGENT RESPONSE:")
            print("="*60)
            print(f"\n{response}\n")
            
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR: {e}")
            print("\n💾 The agent state has been preserved.")
            print("You can restart and your previous context will be loaded.")
            break


def main():
    """Main entry point."""
    try:
        run_cli()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        print("💾 Your session has been saved")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
