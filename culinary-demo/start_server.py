import os
import subprocess

from embed_device import configure_cuda_before_torch

if __name__ == "__main__":
    print("Starting Streamlit App...")
    configure_cuda_before_torch()
    try:
        subprocess.run(["streamlit", "run", "app.py"], check=True, env=os.environ)
    except KeyboardInterrupt:
        print("\nStopping server...")
