"""Main entry point for the culinary agent backend."""

import sys
from pathlib import Path

from agent.core import create_agent
from agent.prompts import SYSTEM_PROMPT
from utils.state_manager import StateManager


def run_cli():
    """Run the CLI interface for the culinary agent."""
    print("="*60)
    print(" 🍳 CULINARY AGENT - Recipe Discovery & Adaptation")
    print("="*60)
    
    # Initialize state manager
    state_manager = StateManager()
    
    # Initialize agent
    print("\nInitializing agent...")
    try:
        agent = create_agent(top_k_recipes=1, max_steps=12)
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
        print("\n📂 Found previous session! Loading last recipe...")
        current_recipe_context = state_manager.load_state()
        if current_recipe_context:
            print(f"RESUMED CONTEXT: {current_recipe_context[:100]}...")
    
    # Get initial user request
    user_request = input("\n🔍 Enter your food request (or 'clear' to start fresh): ").strip()
    
    if user_request.lower() == 'clear':
        current_recipe_context = None
        state_manager.clear_state()
        print("✓ Session cleared")
        user_request = input("Enter new food request: ").strip()
    
    # Main interaction loop
    while True:
        if not user_request:
            print("⚠️  Please enter a valid request")
            user_request = input("\n🔍 Enter your food request: ").strip()
            continue
        
        # Construct prompt
        if not current_recipe_context:
            full_prompt = f"{SYSTEM_PROMPT}\n\nUSER REQUEST: {user_request}"
        else:
            full_prompt = f"""
{SYSTEM_PROMPT}

CONTEXT: Previous recipe provided.
PREVIOUS RECIPE: {current_recipe_context}
USER FEEDBACK: {user_request}
TASK: Adapt PREVIOUS RECIPE to USER FEEDBACK. Validate it.
"""
        
        print("\n" + "="*60)
        print("🤔 Agent is thinking...")
        print("="*60 + "\n")
        
        try:
            # Run the agent
            response = agent.run_with_retry(full_prompt)
            
            # Save state immediately
            current_recipe_context = response
            state_manager.save_state(str(response))
            
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
        
        # Get feedback
        print("-"*60)
        user_request = input("\n💬 Feedback (or type 'exit'/'clear'): ").strip()
        
        if user_request.lower() in ['exit', 'quit', 'q']:
            print("\n👋 Goodbye! Your session has been saved.")
            break
        elif user_request.lower() == 'clear':
            current_recipe_context = None
            state_manager.clear_state()
            print("✓ Session cleared")
            user_request = input("\n🔍 Enter new food request: ").strip()


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