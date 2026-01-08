"""State management for conversation sessions."""

import os
from pathlib import Path
from typing import Optional


class StateManager:
    """Manages conversation state persistence."""
    
    def __init__(self, state_file: Path = Path("last_recipe_state.txt")):
        """Initialize the state manager.
        
        Args:
            state_file: Path to the state persistence file
        """
        self.state_file = state_file if isinstance(state_file, Path) else Path(state_file)
    
    def save_state(self, state: str) -> None:
        """Save the current state to disk.
        
        Args:
            state: State content to save
        """
        with open(self.state_file, "w") as f:
            f.write(state)
    
    def load_state(self) -> Optional[str]:
        """Load the previous state from disk.
        
        Returns:
            Previous state content or None if no state exists
        """
        if not self.state_file.exists():
            return None
        
        with open(self.state_file, "r") as f:
            return f.read()
    
    def clear_state(self) -> None:
        """Clear the saved state."""
        if self.state_file.exists():
            os.remove(self.state_file)
    
    def has_state(self) -> bool:
        """Check if a saved state exists.
        
        Returns:
            True if state exists, False otherwise
        """
        return self.state_file.exists()