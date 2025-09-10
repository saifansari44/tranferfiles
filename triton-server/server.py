# simple_server.py
import subprocess
import sys
import time
import requests
import signal
from pathlib import Path

def start_triton_server(models_path="model_repository"):
    """Start Triton server with simple configuration"""
    
    models_path = Path(models_path).resolve()
    
    # Validate model exists
    if not models_path.exists():
        print(f"Error: Models directory {models_path} not found")
        return False
    
    print(f"Starting Triton server...")
    print(f"Models directory: {models_path}")
    
    # Simple Docker command
    docker_cmd = [
        "docker", "run", "--rm", 
        "-p", "8000:8000",  # HTTP port
        "-p", "8001:8001",  # gRPC port
        "-v", f"{models_path}:/models",
        "nvcr.io/nvidia/tritonserver:24.09-py3",
        "tritonserver", 
        "--model-repository=/models",
        "--model-control-mode=explicit",
        "--log-verbose=1"
    ]

    # docker_cmd = [
    #     "docker", "run", "--rm", 
    #     "-p", "8000:8000",  # HTTP port
    #     "-p", "8001:8001",  # gRPC port
    #     "-v", f"{models_path}:/models",
    #     "my-triton-python-backend:latest",
    #     "tritonserver", 
    #     "--model-repository=/models",
    #     "--model-control-mode=explicit",
    #     "--log-verbose=1"
    # ]
    
    try:
        print("Starting Docker container...")
        process = subprocess.Popen(docker_cmd)
        
        # Handle Ctrl+C
        def signal_handler(sig, frame):
            print("\nStopping server...")
            process.terminate()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Wait for server to be ready
        print("Waiting for server to start...")
        for i in range(30000): # Wait up to 30 seconds
            try:
                response = requests.get("http://localhost:8000/v2/health/ready", timeout=2)
                if response.status_code == 200:
                    print("✓ Server is ready!")
                    print("  HTTP endpoint: http://localhost:8000")
                    print("  gRPC endpoint: localhost:8001")
                    break
            except:
                pass
            time.sleep(1)
        else:
            print("Server startup timeout")
            process.terminate()
            return False
        
        print("Press Ctrl+C to stop the server")
        process.wait()
        
    except Exception as e:
        print(f"Error starting server: {e}")
        return False

if __name__ == "__main__":
    start_triton_server()
