#!/usr/bin/env python3

import os
import sys
import argparse
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_token():
    """Test if the token is valid by trying to download a small file from the model"""
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable is not set")
        return False
        
    print(f"Token length: {len(token)}")
    print(f"Token starts with: {token[:8]}...")
    print(f"Token ends with: ...{token[-4:]}")
    
    try:
        # Try to access cached file first to avoid redundant downloads
        print("Checking token access...")
        
        try:
            # First try to access the cached file without downloading
            cached_file = hf_hub_download(
                repo_id="pyannote/speaker-diarization",
                filename="config.yaml",
                token=token,
                local_files_only=True  # Only check cache, don't download
            )
            print(f"Token test successful - using cached config: {cached_file}")
            return True
        except Exception:
            # File not in cache, download it once to verify access
            print("Config not in cache, downloading for token verification...")
            file = hf_hub_download(
                repo_id="pyannote/speaker-diarization",
                filename="config.yaml",
                token=token
            )
            print(f"Token test successful - config cached to: {file}")
            return True
            
    except Exception as e:
        print(f"Failed to verify token: {str(e)}")
        print(f"Exception type: {type(e)}")
        
        # Check if it's an authentication error
        if "401" in str(e) or "403" in str(e):
            print("\nAuthentication failed. Please ensure:")
            print("1. Your token is correct")
            print("2. You've accepted the user agreement at:")
            print("   https://huggingface.co/pyannote/speaker-diarization")
            print("3. You've also accepted the agreement at:")
            print("   https://huggingface.co/pyannote/segmentation")
        return False

def parse_args():
    parser = argparse.ArgumentParser(description="Test Hugging Face token and model access")
    parser.add_argument("--token-test-only", action="store_true",
                        help="Only test the token and exit")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("Testing token...")
    if not test_token():
        print("ERROR: Token verification failed!")
        print("Please check your token and make sure it's valid.")
        print("Also ensure you've accepted the user agreements for the models at huggingface.co")
        sys.exit(1)
    
    if args.token_test_only:
        print("Token test successful - exiting as requested.")
        sys.exit(0)

if __name__ == "__main__":
    main() 