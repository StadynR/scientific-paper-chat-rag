"""
Script to setup and run ngrok tunnel for Ollama.
Run this script on your local machine before deploying to Streamlit Cloud.
"""

import subprocess
import time
import requests
import sys


def check_ollama_running():
    """
    Check if Ollama is running locally.
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def start_ngrok():
    """
    Start ngrok tunnel and return the public URL.
    """
    print("Starting ngrok tunnel...")
    
    try:
        # Start ngrok process
        process = subprocess.Popen(
            ['ngrok', 'http', '11434', '--log=stdout'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for ngrok to initialize
        print("Waiting for ngrok to initialize...")
        time.sleep(3)
        
        # Get tunnel URL
        response = requests.get('http://localhost:4040/api/tunnels')
        data = response.json()
        
        if 'tunnels' in data and len(data['tunnels']) > 0:
            tunnel_url = data['tunnels'][0]['public_url']
            print(f"\nTunnel successfully created!")
            print(f"Public URL: {tunnel_url}")
            print(f"\nAdd this to your Streamlit Cloud secrets:")
            print(f"OLLAMA_BASE_URL = \"{tunnel_url}\"")
            print(f"\nPress Ctrl+C to stop the tunnel when done.")
            
            # Keep running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping tunnel...")
                process.terminate()
                process.wait()
                print("Tunnel stopped.")
        else:
            print("Error: Could not retrieve tunnel URL")
            process.terminate()
            sys.exit(1)
            
    except FileNotFoundError:
        print("Error: ngrok not found!")
        print("Please install ngrok:")
        print("1. Visit https://ngrok.com/download")
        print("2. Download and install ngrok")
        print("3. Sign up for a free account and get your auth token")
        print("4. Run: ngrok authtoken YOUR_TOKEN")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


def main():
    """
    Main function to setup tunnel.
    """
    print("=" * 60)
    print("Ollama Tunnel Setup for Streamlit Cloud")
    print("=" * 60)
    print()
    
    # Check if Ollama is running
    print("Checking if Ollama is running...")
    if not check_ollama_running():
        print("Error: Ollama is not running!")
        print("Please start Ollama first:")
        print("- On Mac/Linux: ollama serve")
        print("- On Windows: Start Ollama from the app")
        sys.exit(1)
    
    print("Ollama is running.")
    print()
    
    # Start ngrok
    start_ngrok()


if __name__ == "__main__":
    main()
