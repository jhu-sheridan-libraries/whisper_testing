#!/usr/bin/env python3

import os
import sys
import argparse
import whisper

def parse_args():
    parser = argparse.ArgumentParser(description="Download Whisper models")
    parser.add_argument("--model", type=str, default="tiny", 
                      choices=["tiny", "base", "small", "medium", "large", "large-v2"],
                      help="Whisper model size to download")
    parser.add_argument("--output-dir", type=str, default="./models/whisper",
                      help="Directory to save the downloaded model")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Downloading Whisper model: {args.model}")
    print(f"This will be saved to: {args.output_dir}")
    
    try:
        # Check if model already exists
        model_path = os.path.join(args.output_dir, args.model + ".pt")
        if os.path.exists(model_path):
            print(f"Model already exists at {model_path}")
            print("Skipping download. Delete the file if you want to re-download.")
            return
        
        # Download the model
        model = whisper.load_model(
            args.model,
            download_root=args.output_dir,
            in_memory=False  # Ensure model is saved to disk
        )
        
        # Verify the model was saved
        if os.path.exists(model_path):
            print(f"Model {args.model} successfully downloaded to {args.output_dir}")
            print("You can now use this model without downloading it again.")
        else:
            print("Warning: Model download completed but file not found in expected location")
            print(f"Expected at: {model_path}")
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 