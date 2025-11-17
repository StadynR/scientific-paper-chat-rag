"""
Script to run Streamlit app with ngrok tunnel for remote access.
This allows your professor to access the app from anywhere.
"""
import subprocess
import time
import requests
import os
import sys
from pathlib import Path


def check_ollama_running():
    """
    Check if Ollama is running locally.
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def check_ngrok_installed():
    """
    Check if ngrok is installed and configured.
    """
    try:
        result = subprocess.run(
            ['ngrok', 'version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except:
        return False


def get_ngrok_url():
    """
    Get the public ngrok URL for Streamlit.
    """
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get('http://localhost:4040/api/tunnels', timeout=5)
            data = response.json()
            
            if 'tunnels' in data and len(data['tunnels']) > 0:
                for tunnel in data['tunnels']:
                    # Look for the Streamlit tunnel (port 8501)
                    if '8501' in tunnel.get('config', {}).get('addr', ''):
                        return tunnel['public_url']
                
                # If we didn't find 8501 specifically, return the first https tunnel
                for tunnel in data['tunnels']:
                    if tunnel['public_url'].startswith('https'):
                        return tunnel['public_url']
            
            time.sleep(2)
        except:
            time.sleep(2)
    
    return None


def start_streamlit():
    """
    Start Streamlit app.
    """
    print("Starting Streamlit app...")
    
    try:
        # Start Streamlit without capturing output (let it run freely)
        process = subprocess.Popen(
            ['streamlit', 'run', 'app.py', '--server.port=8501'],
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
        )
        
        # Wait for Streamlit to start
        time.sleep(8)
        
        return process
    except Exception as e:
        print(f"Error starting Streamlit: {str(e)}")
        return None


def start_ngrok_tunnel():
    """
    Start ngrok tunnel for Streamlit.
    """
    print("Starting ngrok tunnel...")
    
    try:
        # Start ngrok without capturing output
        process = subprocess.Popen(
            ['ngrok', 'http', '8501'],
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
        )
        
        # Wait for ngrok to initialize
        time.sleep(5)
        
        return process
    except Exception as e:
        print(f"Error starting ngrok: {str(e)}")
        return None


def print_instructions(public_url):
    """
    Print instructions for accessing the app.
    """
    print("\n" + "=" * 70)
    print("Streamlit App is Ready!")
    print("=" * 70)
    print(f"\nPublic URL: {public_url}")
    print("\nTips:")
    print("   - The URL will remain active as long as this script is running")
    print("   - Local access still available at: http://localhost:8501")
    print("\nImportant:")
    print("   - Keep this terminal window open")
    print("   - Keep Ollama running in the background")
    print("   - Press Ctrl+C to stop the tunnel and close the app")
    print("\n" + "=" * 70 + "\n")


def main():
    """
    Main function to run Streamlit with ngrok.
    """
    print("\n" + "=" * 70)
    print("Academic PDF Chat - Streamlit with ngrok")
    print("=" * 70 + "\n")
    
    # Check if Ollama is running
    print("Checking if Ollama is running...")
    if not check_ollama_running():
        print("Error: Ollama is not running!")
        print("\n  Please start Ollama first:")
        print("  - Look for Ollama icon in your system tray")
        print("  - Or start it from the Start menu")
        sys.exit(1)
    print("  Ollama is running.\n")
    
    # Check if ngrok is installed
    print("Checking if ngrok is installed...")
    if not check_ngrok_installed():
        print("Error: ngrok not found!")
        print("\n  Please install ngrok:")
        print("  1. Visit https://ngrok.com/download")
        print("  2. Download and install ngrok")
        print("  3. Sign up for a free account")
        print("  4. Run: ngrok authtoken YOUR_TOKEN")
        sys.exit(1)
    print("  ngrok is installed.\n")
    
    # Start Streamlit
    streamlit_process = start_streamlit()
    if streamlit_process is None:
        print("Failed to start Streamlit")
        sys.exit(1)
    print("  Streamlit started.\n")
    
    # Start ngrok tunnel
    ngrok_process = start_ngrok_tunnel()
    if ngrok_process is None:
        print("Failed to start ngrok tunnel")
        streamlit_process.terminate()
        sys.exit(1)
    print("  ngrok tunnel started.\n")
    
    # Get public URL
    print("Getting public URL...")
    public_url = get_ngrok_url()
    if public_url is None:
        print("Could not retrieve public URL")
        print("  ngrok might still be initializing. Check http://localhost:4040")
        streamlit_process.terminate()
        ngrok_process.terminate()
        sys.exit(1)
    
    # Print instructions
    print_instructions(public_url)
    
    # Keep running
    try:
        while True:
            # Check if processes are still running
            if streamlit_process.poll() is not None:
                print("Streamlit process stopped")
                break
            if ngrok_process.poll() is not None:
                print("ngrok process stopped")
                break
            
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nStopping services...")
    finally:
        # Clean up
        print("  Stopping Streamlit...")
        streamlit_process.terminate()
        try:
            streamlit_process.wait(timeout=5)
        except:
            streamlit_process.kill()
        
        print("  Stopping ngrok...")
        ngrok_process.terminate()
        try:
            ngrok_process.wait(timeout=5)
        except:
            ngrok_process.kill()
        
        print("\nAll services stopped. Goodbye!")


if __name__ == "__main__":
    main()
