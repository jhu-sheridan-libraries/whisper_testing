#!/usr/bin/env python3
import argparse
import os
import torch
from typing import Optional, Tuple, List
import numpy as np
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import webvtt
from tqdm import tqdm
import huggingface_hub
import sys
import time
import csv
import datetime
from pydub import AudioSegment  # For getting audio duration
import psutil  # Add this import at the top

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

def check_hf_token():
    """Verify the Hugging Face token works"""
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("WARNING: HF_TOKEN environment variable not set")
        return False
    try:
        # Test the token by trying to access user info
        user_info = huggingface_hub.whoami(token=token)
        print(f"Authenticated as: {user_info.get('name', 'Unknown')}")
        return True
    except Exception as e:
        print(f"Failed to authenticate with Hugging Face: {str(e)}")
        return False

def transcribe_audio(audio_path: str, model_size: str, task: str, language: Optional[str], num_speakers: Optional[int] = 1) -> List[dict]:
    """Transcribe audio using Whisper"""
    if not os.path.exists(audio_path):
        raise RuntimeError(f"Audio file not found: {audio_path}")
        exit(1)
    
    print(f"Loading Whisper model: {model_size}")
    
    # Set up model paths
    whisper_cache_dir = "/app/models/whisper"
    model_path = os.path.join(whisper_cache_dir, model_size + ".pt")
    
    # Adjust VAD parameters for single speaker
    vad_params = {
        'min_silence_duration_ms': 1000 if num_speakers == 1 else 500,  # More aggressive for single speaker
        'speech_pad_ms': 100 if num_speakers == 1 else 200  # Less padding needed for single speaker
    }
    
    # For tiny model, prefer faster-whisper as it's optimized for speed
    if model_size == "tiny":
        try:
            model_id = f"guillaumekln/faster-whisper-{model_size}"
            print(f"Loading optimized faster-whisper model: {model_id}")
            
            model = WhisperModel(
                model_id,
                device="cpu", 
                compute_type="int8",  # Use int8 quantization for better CPU performance
                download_root=whisper_cache_dir,
                cpu_threads=psutil.cpu_count(logical=True)  # Use all available CPU threads
            )
            
            print(f"Transcribing {audio_path}...")
            segments, info = model.transcribe(
                audio_path, 
                task=task,
                language=language,
                vad_filter=True,
                vad_parameters=vad_params
            )
            
            print(f"Detected language: {info.language} with probability {info.language_probability:.2f}")
            return list(segments)
            
        except Exception as e:
            print(f"Error using faster-whisper: {str(e)}")
            print("Falling back to OpenAI whisper...")
    
    # For other models or if faster-whisper fails, use OpenAI whisper
    try:
        import whisper
        print("Using OpenAI's whisper package")
        
        # Create cache directory if it doesn't exist
        os.makedirs(whisper_cache_dir, exist_ok=True)
        
        if os.path.exists(model_path):
            print(f"Found cached model at {model_path}")
        else:
            print("Model not found in cache, downloading...")
            
        model = whisper.load_model(
            model_size,
            download_root=whisper_cache_dir,
            in_memory=False  # Ensure model is saved to disk
        )
        
        print(f"Transcribing {audio_path} with OpenAI's whisper...")
        result = model.transcribe(
            audio_path,
            task=task,
            language=language,
        )
        
        # Convert whisper format to our format
        segments_list = []
        for segment in result["segments"]:
            segments_list.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })
        return segments_list
    except ImportError:
        print("OpenAI's whisper package not found, falling back to faster-whisper...")
    except Exception as e:
        print(f"Error using OpenAI's whisper: {str(e)}")
        print("Falling back to faster-whisper...")
    
    # Fallback to faster-whisper
    try:
        # Use the correct model ID format for faster-whisper
        model_id = f"guillaumekln/faster-whisper-{model_size}"
        print(f"Loading faster-whisper model: {model_id}")
        
        model = WhisperModel(
            model_id,
            device="cpu", 
            compute_type="int8",
            download_root=whisper_cache_dir
        )
        
        print(f"Transcribing {audio_path}...")
        segments, info = model.transcribe(
            audio_path, 
            task=task,
            language=language,
            vad_filter=True,
            vad_parameters=vad_params
        )
        
        print(f"Detected language: {info.language} with probability {info.language_probability:.2f}")
        
        # Convert generator to list for multiple use
        segments_list = list(segments)
        return segments_list
    except Exception as e:
        print(f"Fatal error loading or using model: {str(e)}")
        raise RuntimeError(f"Could not transcribe audio: {str(e)}")

def perform_diarization(audio_path: str, num_speakers: Optional[int] = None) -> Optional[dict]:
    """Perform speaker diarization using pyannote.audio"""
    # Skip diarization for single speaker
    if num_speakers == 1:
        print("Skipping diarization for single speaker audio")
        return None
        
    print("Loading diarization pipeline...")
    try:
        # Verify token is available
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("ERROR: HF_TOKEN not found in environment")
            return None
            
        print("Attempting to load pipeline with token...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=token
        )
        
        print("Pipeline loaded successfully, setting device...")
        # Set to CPU
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
        print(f"ERROR: Failed to load or run diarization pipeline: {str(e)}")
        print("\nDiarization requires accepting the user agreement at:")
        print("1. https://huggingface.co/pyannote/speaker-diarization")
        print("2. https://huggingface.co/pyannote/segmentation")
        print(f"\nFull error: {str(e)}")
        print(f"Token used: {token[:8]}...")  # Only show first 8 chars
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
        speaker_info = ""
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # If the segment has significant overlap with this speaker turn
            if (max(segment.start, turn.start) < min(segment.end, turn.end)):
                if speaker_info and speaker not in speaker_info:
                    speaker_info += f"+{speaker}"
                else:
                    speaker_info = speaker
        
        if speaker_info:
            segment_dict["speaker"] = speaker_info
        
        enhanced_segments.append(segment_dict)
    
    return enhanced_segments

def save_as_vtt(segments: List[dict], output_path: str):
    """Save the transcription as WebVTT file"""
    vtt = webvtt.WebVTT()
    
    for segment in segments:
        caption = webvtt.Caption(
            start=format_timestamp(segment["start"]),
            end=format_timestamp(segment["end"]),
            text=f"<v {segment.get('speaker', 'UNKNOWN')}>{segment['text']}</v>"
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
    
    # Transcribe the audio with num_speakers parameter
    segments = transcribe_audio(args.audio_file, args.model, args.task, args.language, args.num_speakers)
    print(f"Transcription complete: {len(segments)} segments")
    
    # Try speaker diarization only if more than one speaker
    if args.num_speakers == 1:
        print("Single speaker mode: Using simplified processing")
        enhanced_segments = []
        for segment in segments:
            enhanced_segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "speaker": "SPEAKER_01"  # Single speaker label
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
                    enhanced_segments.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["text"],
                        "speaker": "UNKNOWN"  # Default speaker label
                    })
        except Exception as e:
            print(f"ERROR in diarization: {str(e)}")
            print("Proceeding with transcription only")
            enhanced_segments = []
            for segment in segments:
                enhanced_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                    "speaker": "UNKNOWN"  # Default speaker label
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
                f.write(f"[{segment.get('speaker', 'UNKNOWN')}] {segment['text']}\n")
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

if __name__ == "__main__":
    main()