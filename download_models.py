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
        # Create a temporary directory in the user's home
        temp_dir = os.path.join(os.environ.get('HOME', '/tmp'), '.hf_test')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Try to download the config file using the official API
        print("Attempting to download config file...")
        print(f"Using temporary directory: {temp_dir}")
        
        file = hf_hub_download(
            repo_id="pyannote/speaker-diarization",
            filename="config.yaml",
            token=token,
            local_dir=temp_dir,
            local_dir_use_symlinks=False
        )
        print(f"Successfully downloaded config to: {file}")
        
        # Clean up
        try:
            os.remove(file)
            os.rmdir(temp_dir)
        except:
            pass
            
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