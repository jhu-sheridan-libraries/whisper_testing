#!/usr/bin/env python3
import argparse
import os
import torch
from typing import Optional, Tuple, List
import numpy as np
from faster_whisper import WhisperModel
from huggingface_hub import HfFolder
from pyannote.audio import Pipeline
import webvtt
from tqdm import tqdm
import sys
import time
import csv
import datetime
from pydub import AudioSegment  # For getting audio duration
import psutil  # Add this import at the top

from dotenv import load_dotenv
load_dotenv()

DEFAULT_SPEAK_LABEL = os.environ.get("DEFAULT_SPEAK_LABEL")
BEAM_SIZE = os.environ.get("BEAM_SIZE")
CPU_THRESHOLD = os.environ.get("CPU_THRESHOLD")
MIN_WORKERS = os.environ.get("MIN_WORKERS")

# Add defaults for environment variables
def get_env_float(name, default_value=0.75):
    """Get environment variable as float with default value"""
    value = os.environ.get(name)
    if value:
        try:
            return float(value)
        except ValueError:
            print(f"Warning: Invalid {name} value '{value}'. Using default {default_value}")
    return default_value

def get_env_int(name, default_value=2):
    """Get environment variable as integer with default value"""
    value = os.environ.get(name)
    if value:
        try:
            return int(value)
        except ValueError:
            print(f"Warning: Invalid {name} value '{value}'. Using default {default_value}")
    return default_value

def setup_args():
    parser = argparse.ArgumentParser(description="Transcribe audio with Whisper and perform speaker diarization")
    parser.add_argument("audio_file", type=str, help="Path to the audio file")
    parser.add_argument("--output", type=str, help="Path to output file", default=None)
    parser.add_argument("--model", type=str, default="medium", 
                        choices=["tiny", "base", "small", "medium", "large", "large-v2"],
                        help="Whisper model size")
    parser.add_argument("--format", type=str, default="vtt", choices=["vtt", "srt", "txt"],
                        help="Output format")
    parser.add_argument("--language", type=str, default=None, 
                        help="Language code (transcribe in specified language)")
    parser.add_argument("--task", type=str, default="transcribe", 
                        choices=["transcribe", "translate"],
                        help="Task (transcribe or translate to English)")
    parser.add_argument("--num-speakers", type=int, default=None,
                        help="Number of speakers expected in the audio (improves diarization)")
    return parser.parse_args()


def get_hf_token():
    """Check env, if in colab use colab env, otherwise use sys env for HF token"""
    try:
        import google.colab
        token = HfFolder().get_token()
        if not token:
            raise ValueError(
                "Hugging Face token not found. Please run `notebook_login()` in Colab."
            )
        return token
    except ImportError:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise EnvironmentError(
                "HF_TOKEN environment variable not set. "
                "Please set it in your .env file or your shell environment."
            )
        return token


def check_hf_token():
    """Verify the Hugging Face token works and has necessary model access"""
    token = get_hf_token()

    print("Configuring Hugging Face credentials...")
    print("Testing Hugging Face token...")
    print("Testing token...")
    print(f"Token length: {len(token)}")
    print(f"Token starts with: {token[:8]}...")
    print(f"Token ends with: ...{token[-4:]}")

    try:
        from huggingface_hub import HfApi
        api = HfApi()

        # List of models we need access to
        required_models = [
            {
                "model_id": "pyannote/speaker-diarization-3.1",
                "name": "Speaker Diarization Pipeline 3.1",
                "license_url": "https://huggingface.co/pyannote/speaker-diarization-3.1"
            },
            {
                "model_id": "pyannote/segmentation-3.0",
                "name": "Speaker Segmentation Model 3.0",
                "license_url": "https://huggingface.co/pyannote/segmentation-3.0"
            },
            # Fallback options for different versions
            {
                "model_id": "pyannote/speaker-diarization",
                "name": "Speaker Diarization Pipeline (legacy)",
                "license_url": "https://huggingface.co/pyannote/speaker-diarization" 
            },
            {
                "model_id": "pyannote/segmentation",
                "name": "Speaker Segmentation Model (legacy)",
                "license_url": "https://huggingface.co/pyannote/segmentation"
            }
        ]

        print("\nChecking model access permissions:")
        all_access = True
        
        for model in required_models:
            try:
                # Try to get model info - will fail if no access
                model_info = api.model_info(model["model_id"], token=token)
                print(f"\n✓ Access GRANTED to {model['name']}")
                print(f"  Model ID: {model['model_id']}")
                if hasattr(model_info, 'cardData'):
                    if 'license' in model_info.cardData:
                        print(f"  License: {model_info.cardData['license']}")
            except Exception as e:
                all_access = False
                print(f"\n✗ Access DENIED to {model['name']}")
                print(f"  Model ID: {model['model_id']}")
                print(f"  Error: {str(e)}")
                print(f"  To fix:")
                print(f"  1. Visit {model['license_url']}")
                print(f"  2. Log in with your Hugging Face account")
                print(f"  3. Accept the license agreement")
                print(f"  4. Wait a few minutes for permissions to propagate")

        if not all_access:
            print("\n⚠️  Some model access is missing!")
            print("Please accept the license agreements for the models listed above.")
            print("After accepting the agreements, wait a few minutes and try again.")
            print("\nIf you continue to have issues:")
            print("1. Try logging out and back in to Hugging Face")
            print("2. Generate a new token at https://hf.co/settings/tokens")
            print("3. Update your .env file with the new token")
            return False

        print("\n✓ All required model access verified!")
        return True

    except Exception as e:
        print(f"\nError verifying token access: {str(e)}")
        return False

def transcribe_audio(audio_path: str, model_size: str, task: str, language: Optional[str], num_speakers: Optional[int] = 1) -> List[dict]:
    """Transcribe audio using Whisper with auto-detection for compute device and type."""
    if not os.path.exists(audio_path):
        raise RuntimeError(f"Audio file not found: {audio_path}")
        exit(1)

    print(f"Loading Whisper model: {model_size}")

    # --- Determine Device and Compute Type --- xxx
    precision_hint = os.environ.get("PRECISION", "auto").lower()
    device = "cpu"
    compute_type = "int8" # Default fallback

    if torch.cuda.is_available():
        print("CUDA GPU detected. Trying to use GPU.")
        device = "cuda"
        if precision_hint == "float16" or precision_hint == "auto":
            compute_type = "float16"
            print("Using compute type: float16 (recommended for GPU)")
        elif precision_hint == "int8":
            compute_type = "int8"
            print("Using compute type: int8 (as requested)")
        elif precision_hint == "float32":
             compute_type = "float32"
             print("Using compute type: float32 (as requested, might be slower on GPU)")
        else:
            compute_type = "float16" # Default for GPU if hint is invalid
            print(f"Invalid PRECISION hint '{precision_hint}'. Defaulting to compute type: float16 for GPU")
    else:
        print("No CUDA GPU detected. Using CPU.")
        device = "cpu"
        if precision_hint == "int8" or precision_hint == "auto":
            compute_type = "int8"
            print("Using compute type: int8 (recommended for CPU)")
        elif precision_hint == "float32":
            compute_type = "float32"
            print("Using compute type: float32 (might be slow on CPU)")
        elif precision_hint == "float16":
            compute_type = "int8" # float16 not ideal on CPU, use int8
            print("PRECISION hint 'float16' not recommended for CPU. Using compute type: int8")
        else:
            compute_type = "int8" # Default for CPU if hint is invalid
            print(f"Invalid PRECISION hint '{precision_hint}'. Defaulting to compute type: int8 for CPU")
    # --- End Determine Device and Compute Type ---

    # --- Determine Beam Size ---
    beam_size_int = get_env_int("BEAM_SIZE", 5)  # Default beam size is 5
    print(f"Using beam size: {beam_size_int}")
    # --- End Determine Beam Size ---

    # Set up model paths
    whisper_cache_dir = "/app/models/whisper"
    # Model path for OpenAI whisper isn't directly used by faster-whisper loading
    # model_path = os.path.join(whisper_cache_dir, model_size + ".pt")

    # Adjust VAD parameters based on speaker count (remains unchanged)
    vad_params = {
        'min_silence_duration_ms': 1000 if num_speakers == 1 else 500,
        'speech_pad_ms': 100 if num_speakers == 1 else 200
    }

    # Try faster-whisper first, as it's generally preferred
    try:
        # Use the correct model ID format for faster-whisper, adjusted for v3 if applicable
        # Mapping common names to potential faster-whisper names
        model_repo_map = {
            "large-v2": "large-v2",
            "large-v3": "large-v3", # Assuming a large-v3 exists or will exist
            "large": "large-v2", # Default large to v2 for now
            "medium": "medium",
            "small": "small",
            "base": "base",
            "tiny": "tiny",
            "medium.en": "medium.en",
            "small.en": "small.en",
            "base.en": "base.en",
            "tiny.en": "tiny.en",
        }
        faster_model_size = model_repo_map.get(model_size, model_size) # Fallback to original name if not in map
        # Check if it's a distilled model
        is_distilled = "distil" in faster_model_size
        if is_distilled:
             # Distilled models often have specific repo names
             # Example: 'Systran/faster-distil-whisper-large-v2'
             # We might need a more robust way to map these, but let's try a convention
             # Assuming the user might provide 'distil-large-v2' etc.
             # We'll prepend 'Systran/faster-' if it seems like a distilled model name
             if not faster_model_size.startswith("Systran/faster-"):
                 model_id = f"Systran/faster-{faster_model_size}"
             else:
                 model_id = faster_model_size
        else:
             # Standard faster-whisper models from guillaumekln
             model_id = f"guillaumekln/faster-whisper-{faster_model_size}"


        print(f"Attempting to load faster-whisper model: {model_id} with device: {device}, compute_type: {compute_type}")

        # Determine optimal thread and worker count
        total_cpu_threads = psutil.cpu_count(logical=True)
        physical_cores = psutil.cpu_count(logical=False)
        
        # For CPU device, use thread configuration
        if device == "cpu":
            # Use specified percentage of available logical cores for processing
            cpu_threshold = get_env_float("CPU_THRESHOLD", 0.75)
            optimal_threads = max(1, int(total_cpu_threads * cpu_threshold))
            print(f"CPU Mode: Using {optimal_threads} threads out of {total_cpu_threads} available (using {cpu_threshold:.2f} of capacity)")
            
            # For num_workers, use up to 1/4 of physical cores, minimum 1, maximum 4
            min_workers = get_env_int("MIN_WORKERS", 2)
            optimal_workers = min(4, max(min_workers, physical_cores // 4))
            print(f"Setting {optimal_workers} worker processes for data loading")
        else:
            # For GPU, we don't need as many CPU resources
            optimal_threads = physical_cores  # Use physical cores count for CPU ops
            min_workers = get_env_int("MIN_WORKERS", 2)
            optimal_workers = min_workers
            print(f"GPU Mode: Using {optimal_threads} threads for CPU operations")

        model = WhisperModel(
            model_id,
            device=device,
            compute_type=compute_type,
            download_root=whisper_cache_dir,
            cpu_threads=optimal_threads,
            num_workers=optimal_workers
        )

        print(f"Transcribing {audio_path} using faster-whisper...")
        segments_generator, info = model.transcribe(
            audio_path,
            task=task,
            language=language,
            vad_filter=True,
            vad_parameters=vad_params,
            beam_size=beam_size_int
        )

        print(f"Detected language: {info.language} with probability {info.language_probability:.2f}\n")

        print("Converting generator to list for subsequent processing...")
        # Convert faster-whisper segments to our common dictionary format
        segments_list = []
        for segment in segments_generator:
            segments_list.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            })
        print(f"Faster-whisper transcription successful: {len(segments_list)} segments found.")
        return segments_list

    except ImportError:
        print("faster-whisper package not found or import failed.")
        pass

    except Exception as e:
        print(f"Error loading or using faster-whisper model '{model_id}': {str(e)}")
        print("Attempting to fall back to OpenAI's whisper package...")

    # Fallback to OpenAI's whisper package
    try:
        import whisper
        print("Using OpenAI's whisper package as fallback.")

        # Create cache directory if it doesn't exist
        os.makedirs(whisper_cache_dir, exist_ok=True)

        # OpenAI whisper handles device selection automatically ('cuda' if available, else 'cpu')
        print(f"Loading OpenAI whisper model: {model_size} (device auto-selected)")
        # Note: download_root might behave differently or might need model path check
        model_path = os.path.join(whisper_cache_dir, model_size + ".pt")
        if os.path.exists(model_path):
             print(f"Found cached OpenAI model at {model_path}")
        else:
             print("OpenAI Model not found in cache, will download if needed...")

        # Load model - device selection is automatic here
        model = whisper.load_model(
            model_size,
            download_root=whisper_cache_dir,
            in_memory=False # Keep False to ensure caching works as expected
        )

        print(f"Transcribing {audio_path} with OpenAI's whisper...")
        result = model.transcribe(
            audio_path,
            task=task,
            language=language,
        )

        # Convert OpenAI whisper format to our common format
        segments_list = []
        for segment in result["segments"]:
            segments_list.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })
        print(f"OpenAI whisper transcription successful: {len(segments_list)} segments found.")
        return segments_list

    except ImportError:
        print("FATAL: OpenAI's whisper package not found, and faster-whisper failed or is unavailable.")
        raise RuntimeError("Neither faster-whisper nor OpenAI whisper could be used.")
    except Exception as e:
        print(f"FATAL: Error using OpenAI's whisper: {str(e)}")
        raise RuntimeError(f"Could not transcribe audio using either library: {str(e)}")

def perform_diarization(audio_path: str, num_speakers: Optional[int] = None) -> Optional[dict]:
    """Perform speaker diarization using pyannote.audio"""
    # Skip diarization for single speaker
    if num_speakers == 1:
        print("Skipping diarization for single speaker audio")
        return None
        
    print("Loading diarization pipeline...")
    try:
        # Verify token is available and valid
        token = get_hf_token()
        if not token:
            print("ERROR: HF_TOKEN not found in environment")
            print("Please set your Hugging Face token in the .env file")
            print("Visit https://hf.co/settings/tokens to create one")
            return None

        # Test token validity with specific model access
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            
            # Check access to required models
            models_to_check = [
                "pyannote/speaker-diarization-3.1",
                "pyannote/segmentation-3.0",
                "pyannote/speaker-diarization",
                "pyannote/segmentation"
            ]
            
            print("\nVerifying access to required models...")
            for model in models_to_check:
                try:
                    # Try to get model info - will fail if no access
                    api.model_info(model, token=token)
                    print(f"✓ Access verified for {model}")
                except Exception as e:
                    print(f"✗ Cannot access {model}")
                    print(f"  Please visit https://huggingface.co/{model}")
                    print("  and accept the user agreement.")
            
            print("\nAttempting to load pipeline...")
            
        except Exception as e:
            print(f"Error verifying token access: {str(e)}")
            
        # Try different pipeline versions in order of preference
        pipeline_models = [
            "pyannote/speaker-diarization-3.1",
            "pyannote/speaker-diarization",
        ]
        
        pipeline = None
        last_error = None
        
        # Try each pipeline model until one works
        for pipeline_model in pipeline_models:
            try:
                print(f"Attempting to load pipeline from {pipeline_model}...")
                pipeline = Pipeline.from_pretrained(
                    pipeline_model,
                    use_auth_token=token
                )
                
                if pipeline is not None:
                    print(f"✓ Successfully loaded {pipeline_model}")
                    break
            except Exception as e:
                print(f"✗ Failed to load {pipeline_model}: {str(e)}")
                last_error = e
        
        # If no pipeline was successfully loaded
        if pipeline is None:
            if last_error:
                raise last_error
            else:
                raise RuntimeError("Failed to load any diarization pipeline")
                
        print("Pipeline loaded successfully, setting device...")
        # Set to CPU - we'll enhance this later to handle GPU if available
        pipeline.to(torch.device("cpu"))
        
        print("Performing speaker diarization...")
        if num_speakers is not None:
            print(f"Using specified number of speakers: {num_speakers}")
            diarization = pipeline(audio_path, num_speakers=num_speakers)
        else:
            print("Automatically determining number of speakers")
            diarization = pipeline(audio_path)
        
        return diarization
            
    except Exception as e:
        print(f"\nERROR: Failed to load or run diarization pipeline: {str(e)}")
        print("\nThis might be because:")
        print("1. Your token doesn't have the required permissions")
        print("2. You haven't accepted the model terms of use")
        print("\nPlease visit and accept terms for ALL of these models:")
        print("1. https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("2. https://huggingface.co/pyannote/segmentation-3.0")
        print("3. https://huggingface.co/pyannote/speaker-diarization")
        print("4. https://huggingface.co/pyannote/segmentation")
        print("\nIf you've already accepted the terms, try:")
        print("1. Logging out and back in to Hugging Face")
        print("2. Generating a new token at https://hf.co/settings/tokens")
        print("3. Updating your .env file with the new token")
        print(f"\nFull error: {str(e)}")
        return None

def combine_transcription_with_diarization(segments: List[dict], diarization) -> List[dict]:
    """Combine transcription segments with speaker information"""
    print("Combining transcription with speaker diarization...")
    
    enhanced_segments = []
    
    for segment in tqdm(segments):
        segment_dict = {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        }
        
        # Find speaker for this time segment
        best_speaker = None
        max_overlap = 0

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            overlap_start = max(segment.start, turn.start)
            overlap_end = min(segment.end, turn.end)
            overlap_duration = overlap_end - overlap_start

            if overlap_duration > 0:
                 if overlap_duration > max_overlap:
                     max_overlap = overlap_duration
                     best_speaker = speaker
                 elif best_speaker and speaker != best_speaker:
                     # Handle segments potentially spanning multiple speakers if needed
                     # For simplicity, maybe just take the first dominant one
                     pass # Keep the one with the largest overlap found so far

        if best_speaker:
            segment_dict["speaker"] = best_speaker
        else:
            segment_dict["speaker"] = DEFAULT_SPEAK_LABEL
        
        enhanced_segments.append(segment_dict)
    
    return enhanced_segments

def save_as_vtt(segments: List[dict], output_path: str):
    """Save the transcription as WebVTT file"""
    vtt = webvtt.WebVTT()
    
    for segment in segments:
        caption = webvtt.Caption(
            start=format_timestamp(segment["start"]),
            end=format_timestamp(segment["end"]),
            text=f"<v {segment.get('speaker', DEFAULT_SPEAK_LABEL)}>{segment['text']}</v>"
        )
        vtt.captions.append(caption)
    
    vtt.save(output_path)
    print(f"Saved WebVTT file to {output_path}")

def format_timestamp(seconds: float) -> str:
    """Format timestamp for WebVTT format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

def get_audio_duration(audio_path: str) -> float:
    """Get duration of audio file in seconds"""
    try:
        audio = AudioSegment.from_file(audio_path)
        return len(audio) / 1000.0  # Convert milliseconds to seconds
    except Exception as e:
        print(f"Warning: Could not get audio duration: {str(e)}")
        return 0.0

def log_benchmark(audio_path: str, duration: float, num_speakers: int, model: str, processing_time: float):
    """Log benchmark data to CSV file"""
    benchmark_file = "/data/whisper_benchmarks.csv"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get CPU info
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    memory_percent = psutil.virtual_memory().percent
    
    # Prepare the header and data
    header = [
        'timestamp', 'filename', 'duration_seconds', 'num_speakers', 
        'model', 'processing_time_seconds', 'real_time_factor',
        'cpu_percent', 'cpu_count', 'memory_percent'
    ]
    data = [
        timestamp, os.path.basename(audio_path), duration, num_speakers, 
        model, processing_time, (processing_time/duration),
        cpu_percent, cpu_count, memory_percent
    ]
    
    # Check if file exists to determine if we need to write header
    file_exists = os.path.exists(benchmark_file)
    
    with open(benchmark_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(data)
    
    # Print benchmark summary
    print("\nBenchmark Summary:")
    print(f"File: {os.path.basename(audio_path)}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Model: {model}")
    print(f"Speakers: {num_speakers if num_speakers else 'auto'}")
    print(f"Processing Time: {processing_time:.2f} seconds")
    print(f"Real-time Factor: {(processing_time/duration):.2f}x")
    print(f"CPU Usage: {cpu_percent}% (of {cpu_count} CPUs)")
    print(f"Memory Usage: {memory_percent}%")
    print(f"Results saved to: {benchmark_file}")

def estimate_processing_time(duration: float, model: str, num_speakers: Optional[int] = None) -> dict:
    """Estimate processing time based on file duration, model size, and speakers"""
    # Base real-time factors for transcription (empirically determined)
    model_factors = {
        "tiny": 1.2,    # ~1.2x real-time
        "base": 2.0,    # ~2.0x real-time
        "small": 3.0,   # ~3.0x real-time
        "medium": 4.5,  # ~4.5x real-time
        "large": 6.0,   # ~6.0x real-time
        "large-v2": 7.0 # ~7.0x real-time
    }
    
    # Diarization adds overhead
    diarization_factor = 1.5 if num_speakers else 0  # ~1.5x additional time if diarization enabled
    
    # Calculate estimates
    transcription_time = duration * model_factors.get(model, 4.5)  # default to medium if unknown
    diarization_time = duration * diarization_factor
    total_time = transcription_time + diarization_time
    
    return {
        "total_minutes": total_time / 60,
        "transcription_minutes": transcription_time / 60,
        "diarization_minutes": diarization_time / 60,
        "real_time_factor": total_time / duration
    }

def print_time_estimate(duration: float, model: str, num_speakers: Optional[int] = None):
    """Print a human-readable time estimate"""
    estimate = estimate_processing_time(duration, model, num_speakers)
    
    print("\nEstimated Processing Time:")
    print(f"Audio Duration: {duration/60:.1f} minutes")
    print(f"Model: {model}")
    print(f"Speakers: {'auto' if num_speakers is None else num_speakers}")
    print(f"Expected total time: {estimate['total_minutes']:.1f} minutes")
    print(f"  - Transcription: {estimate['transcription_minutes']:.1f} minutes")
    if num_speakers:
        print(f"  - Diarization: {estimate['diarization_minutes']:.1f} minutes")
    else:
        print(f"  - Diarization not indicated.")
    print(f"Estimated real-time factor: {estimate['real_time_factor']:.1f}x")
    print("Note: Estimates are approximate and may vary based on CPU speed and audio complexity\n")

def main():
    args = setup_args()
    
    # Get audio duration
    duration = get_audio_duration(args.audio_file)
    print(f"Audio duration: {duration:.2f} seconds")
    
    # Show time estimate
    print_time_estimate(duration, args.model, args.num_speakers)
    
    # Ask for confirmation if it's a long process
    if duration > 600:  # If longer than 10 minutes
        estimate = estimate_processing_time(duration, args.model, args.num_speakers)
        if estimate['total_minutes'] > 30:  # If estimated time > 30 minutes
            response = input(f"\nThis could take {estimate['total_minutes']:.1f} minutes. Continue? (y/n): ")
            if response.lower() != 'y':
                print("Operation cancelled by user")
                sys.exit(0)
    
    # Start timing
    start_time = time.time()
    
    # Validate the output path if provided
    output_path = args.output
    if not output_path:
        # Generate default output path by replacing extension
        input_basename = os.path.splitext(os.path.basename(args.audio_file))[0]
        output_path = f"/data/{input_basename}.{args.format}"
        print(f"No output path specified, using: {output_path}")
    
    # Verify input file exists
    if not os.path.exists(args.audio_file):
        print(f"ERROR: Input file not found: {args.audio_file}")
        sys.exit(1)
    
    try:
        # Transcribe the audio with num_speakers parameter
        segments = transcribe_audio(args.audio_file, args.model, args.task, args.language, args.num_speakers)
        if not segments:
            raise RuntimeError("Transcription failed to return any segments")
            
        print(f"Transcription complete: {len(segments)} segments")
        
        # Try speaker diarization only if more than one speaker
        if args.num_speakers == 1:
            print("Single speaker mode: Using simplified processing")
            enhanced_segments = []
            for segment in segments:
                # Ensure segment is in dictionary format
                if isinstance(segment, (tuple, list)):
                    # Convert tuple/list to dict if needed
                    enhanced_segments.append({
                        "start": float(segment[0]) if len(segment) > 0 else 0.0,
                        "end": float(segment[1]) if len(segment) > 1 else 0.0,
                        "text": str(segment[2]) if len(segment) > 2 else "",
                        "speaker": "SPEAKER_01"
                    })
                else:
                    # Already in dict format
                    enhanced_segments.append({
                        "start": float(segment.get("start", 0.0)),
                        "end": float(segment.get("end", 0.0)),
                        "text": str(segment.get("text", "")),
                        "speaker": "SPEAKER_01"
                    })
        else:
            try:
                diarization = perform_diarization(args.audio_file, num_speakers=args.num_speakers)
                
                if diarization:
                    # Combine transcription with speaker diarization
                    print("Combining transcription with speaker information...")
                    enhanced_segments = combine_transcription_with_diarization(segments, diarization)
                else:
                    # Use plain transcription without speaker info
                    print("WARNING: Proceeding with transcription only (no speaker diarization)")
                    enhanced_segments = []
                    for segment in segments:
                        # Ensure segment is in dictionary format
                        if isinstance(segment, (tuple, list)):
                            enhanced_segments.append({
                                "start": float(segment[0]) if len(segment) > 0 else 0.0,
                                "end": float(segment[1]) if len(segment) > 1 else 0.0,
                                "text": str(segment[2]) if len(segment) > 2 else "",
                                "speaker": DEFAULT_SPEAK_LABEL
                            })
                        else:
                            enhanced_segments.append({
                                "start": float(segment.get("start", 0.0)),
                                "end": float(segment.get("end", 0.0)),
                                "text": str(segment.get("text", "")),
                                "speaker": DEFAULT_SPEAK_LABEL
                            })
            except Exception as e:
                print(f"ERROR in diarization: {str(e)}")
                print("Proceeding with transcription only")
                enhanced_segments = []
                for segment in segments:
                    # Ensure segment is in dictionary format
                    if isinstance(segment, (tuple, list)):
                        enhanced_segments.append({
                            "start": float(segment[0]) if len(segment) > 0 else 0.0,
                            "end": float(segment[1]) if len(segment) > 1 else 0.0,
                            "text": str(segment[2]) if len(segment) > 2 else "",
                            "speaker": DEFAULT_SPEAK_LABEL
                        })
                    else:
                        enhanced_segments.append({
                            "start": float(segment.get("start", 0.0)),
                            "end": float(segment.get("end", 0.0)),
                            "text": str(segment.get("text", "")),
                            "speaker": DEFAULT_SPEAK_LABEL
                        })
        
        # Save the results based on the format
        if args.format == "vtt":
            save_as_vtt(enhanced_segments, output_path)
        elif args.format == "srt":
            # Add SRT output support if needed
            print("SRT format not implemented yet, saving as VTT")
            save_as_vtt(enhanced_segments, output_path)
        elif args.format == "txt":
            # Add plain text output support if needed
            with open(output_path, "w") as f:
                for segment in enhanced_segments:
                    f.write(f"[{segment.get('speaker', DEFAULT_SPEAK_LABEL)}] {segment['text']}\n")
            print(f"Saved plain text file to {output_path}")
        
        # Calculate total processing time and log benchmark
        processing_time = time.time() - start_time
        log_benchmark(
            audio_path=args.audio_file,
            duration=duration,
            num_speakers=args.num_speakers,
            model=args.model,
            processing_time=processing_time
        )
        
        print(f"Processing complete! Output saved to {output_path}")

        # Print time estimate
        print_time_estimate(duration, args.model, args.num_speakers)
        
    except Exception as e:
        print(f"ERROR: Processing failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()