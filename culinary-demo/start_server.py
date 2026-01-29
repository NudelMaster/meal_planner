import subprocess
import os

if __name__ == "__main__":
    print("Starting Streamlit App...")
    # Use array for command to avoid shell injection, though purely local here
    try:
        subprocess.run(["streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\nStopping server...")
