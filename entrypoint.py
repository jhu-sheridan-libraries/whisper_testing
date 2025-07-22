#!/usr/bin/env python3
"""
Cross-platform entrypoint script for whisper transcription container.
Replaces bash entrypoint.sh for Windows/Linux compatibility.
"""
import os
import sys
import subprocess
import platform
from pathlib import Path

def print_env_debug():
    """Print environment variables for debugging"""
    print("Environment variables at startup:")
    for key, value in os.environ.items():
        if 'hf_' in key.lower():
            print(f"{key}={value}")

def debug_mode():
    """Debug information display"""
    print("=== Debug Mode ===")
    print(f"Platform: {platform.system()}")
    print(f"User: {os.getenv('USER', os.getenv('USERNAME', 'unknown'))}")
    print(f"Current directory: {os.getcwd()}")
    
    app_dir = Path("/app")
    if app_dir.exists():
        print("Contents of /app:")
        for item in app_dir.iterdir():
            print(f"  {item.name}")
    
    models_dir = Path("/app/models")
    if models_dir.exists():
        print("Contents of /app/models:")
        for item in models_dir.iterdir():
            print(f"  {item.name}")

def check_cached_models():
    """Check for cached Whisper models"""
    print("Checking for cached models...")
    whisper_cache = Path("/app/models/whisper")
    
    if whisper_cache.exists():
        print(f"Found Whisper cache directory:")
        print(f"Contents of {whisper_cache}:")
        for item in whisper_cache.iterdir():
            if item.is_file():
                size = item.stat().st_size
                print(f"  {item.name} ({size} bytes)")
        
        # Check for specific models
        models = ["tiny.pt", "base.pt", "small.pt", "medium.pt", "large.pt"]
        for model in models:
            model_path = whisper_cache / model
            if model_path.exists():
                print(f"Found cached {model.replace('.pt', '')} model")
    else:
        print("No cached Whisper models found. First run will download models.")
        whisper_cache.mkdir(parents=True, exist_ok=True)

def setup_huggingface_credentials():
    """Configure Hugging Face credentials cross-platform"""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        return
    
    print("Configuring Hugging Face credentials...")
    
    # Get home directory cross-platform
    home_dir = Path.home()
    
    # Create necessary directories
    hf_cache_dir = home_dir / ".cache" / "huggingface"
    hf_dir = home_dir / ".huggingface"
    
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    hf_dir.mkdir(parents=True, exist_ok=True)
    
    # Set appropriate permissions (Unix-like only)
    if platform.system() != "Windows":
        os.chmod(hf_cache_dir, 0o700)
        os.chmod(hf_dir, 0o700)
    
    # Write token to files
    token_file1 = hf_cache_dir / "token"
    token_file2 = hf_dir / "token"
    
    token_file1.write_text(hf_token)
    token_file2.write_text(hf_token)
    
    # Set environment variables
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    os.environ["HF_HUB_TOKEN"] = hf_token
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HF_HOME"] = str(hf_cache_dir)
    
    # Test token
    print("Testing Hugging Face token...")
    try:
        result = subprocess.run([
            sys.executable, "/app/download_models.py", "--token-test-only"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print("WARNING: Token test failed. Continuing anyway, but model downloads may fail.")
            print(f"Error: {result.stderr}")
        else:
            print("Token verified successfully.")
    except subprocess.TimeoutExpired:
        print("WARNING: Token test timed out.")
    except Exception as e:
        print(f"WARNING: Token test failed with exception: {e}")

def main():
    """Main entrypoint logic"""
    args = sys.argv[1:]
    
    # Print environment debug info
    print_env_debug()
    
    # Handle debug mode
    if args and args[0] == "--debug":
        debug_mode()
        sys.exit(0)
    
    # Check cached models
    check_cached_models()
    
    # Setup HuggingFace credentials
    setup_huggingface_credentials()
    
    # Handle different command types
    if not args:
        # Default help
        subprocess.run([sys.executable, "/app/transcribe_diarize.py", "--help"])
        return
    
    # Direct commands
    direct_commands = ["ls", "file", "echo", "cat", "find"]
    if args[0] in direct_commands:
        subprocess.run(args)
        return
    
    # Shell separator
    if args[0] == "--":
        subprocess.run(args[1:])
        return
    
    # Interactive shell
    if args[0] == "--shell":
        if platform.system() == "Windows":
            subprocess.run(["cmd"])
        else:
            subprocess.run(["/bin/bash"])
        return
    
    # Default: run transcription script
    subprocess.run([sys.executable, "/app/transcribe_diarize.py"] + args)

if __name__ == "__main__":
    main()