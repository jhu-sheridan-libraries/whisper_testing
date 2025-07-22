#!/usr/bin/env python3
import argparse
import os
import torch
import signal  # For timeout handling
from typing import Optional, Tuple, List, Iterator, Dict
from faster_whisper import WhisperModel
from huggingface_hub import HfFolder
from pyannote.audio import Pipeline
import webvtt
from tqdm import tqdm
import sys
import time
import csv
import datetime
import psutil  # Add this import at the top
from pathlib import Path  # For cross-platform path handling

from dotenv import load_dotenv
load_dotenv()

DEFAULT_SPEAK_LABEL = os.environ.get("DEFAULT_SPEAK_LABEL", "SPEAKER_UNKNOWN")
BEAM_SIZE = os.environ.get("BEAM_SIZE")
CPU_THRESHOLD = os.environ.get("CPU_THRESHOLD")
MIN_WORKERS = os.environ.get("MIN_WORKERS")

import os
from pydub import AudioSegment


def convert_mp4_to_wav(input_path: str, output_path: str = None) -> str:
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"No such file: {input_path}")

    base, _ext = os.path.splitext(input_path)
    wav_path = output_path or f"{base}.wav"

    audio = AudioSegment.from_file(input_path, format="mp4")
    audio.export(wav_path, format="wav")

    return wav_path


def can_load_mp4_audio(path: str) -> bool:
    """test if torchaudio can load a file, if it can then mp4 is supported"""
    try:
        import torchaudio
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            _waveform, _sr = torchaudio.load(path)
        return True
    except Exception:
        return False


def is_running_in_colab() -> bool:
    try:
        import google.colab  # this module only exists in Colab
        return True
    except ImportError:
        return False

# Add defaults for environment variables
def get_env_str(name: str, default_value: str) -> str:
    """Get environment variable as string with default value"""
    return os.environ.get(name, default_value)

def get_env_float(name: str, default_value: float) -> float:
    """Get environment variable as float with default value"""
    value = os.environ.get(name)
    if value:
        try:
            return float(value)
        except ValueError:
            print(f"Warning: Invalid {name} value '{value}'. Using default {default_value}")
    return default_value

def get_env_int(name: str, default_value: int) -> int:
    """Get environment variable as integer with default value"""
    value = os.environ.get(name)
    if value:
        try:
            return int(value)
        except ValueError:
            print(f"Warning: Invalid {name} value '{value}'. Using default {default_value}")
    return default_value

def timeout_handler(signum, frame):
    raise TimeoutError("Diarization process timed out")

def setup_args():
    parser = argparse.ArgumentParser(description="Transcribe audio with Whisper and perform speaker diarization")
    parser.add_argument("audio_file", type=str, help="Path to the audio file")
    parser.add_argument("--output", type=str, default=get_env_str("OUTPUT_PATH", None), help="Path to output file")
    parser.add_argument("--model", type=str, default=get_env_str("MODEL_SIZE", "medium"), 
                        choices=["tiny", "base", "small", "medium", "large", "large-v2"],
                        help="Whisper model size")
    parser.add_argument("--format", type=str, default=get_env_str("OUTPUT_FORMAT", "vtt"), choices=["vtt", "srt", "txt", "json"],
                        help="Output format")
    parser.add_argument("--language", type=str, default=get_env_str("WHISPER_LANGUAGE", None), 
                        help="Language code (transcribe in specified language)")
    parser.add_argument("--task", type=str, default=get_env_str("TASK", "transcribe"), 
                        choices=["transcribe", "translate"],
                        help="Task (transcribe or translate to English)")
    parser.add_argument("--num-speakers", type=int, default=get_env_int("NUM_SPEAKERS", None),
                        help="Number of speakers expected in the audio (improves diarization)")
    parser.add_argument("--diarization-pipeline", type=str, default=get_env_str("DIARIZATION_PIPELINE", "pyannote/speaker-diarization-3.1"),
                        choices=["pyannote/speaker-diarization-3.1", "pyannote/speaker-diarization"],
                        help="Diarization pipeline version to use")
    parser.add_argument("--transcription-model", type=str, default=get_env_str("TRANSCRIPTION_MODEL", "faster-whisper"),
                        choices=["faster-whisper", "openai-whisper"],
                        help="Transcription model to use")
    parser.add_argument("--cpu-threads", type=int, default=get_env_int("CPU_THREADS", 0),
                        help="Number of CPU threads to use for transcription (0 for optimal)")
    parser.add_argument("--beam-size", type=int, default=get_env_int("BEAM_SIZE", 5),
                        help="Beam size for transcription")
    parser.add_argument("--vad-level", type=int, default=get_env_int("VAD_LEVEL", 2), choices=[0, 1, 2, 3],
                        help="VAD level (0=low, 3=high)")
    parser.add_argument("--num-workers", type=int, default=get_env_int("NUM_WORKERS", 0),
                        help="Number of workers to use for transcription (0 for optimal)")
    parser.add_argument("--precision", type=str, default=get_env_str("PRECISION", "auto"), choices=["auto", "float16", "int8", "float32"],
                        help="Model precision")
    parser.add_argument("--gpu-device", type=int, default=get_env_int("GPU_DEVICE", 0),
                        help="GPU device to use for transcription")
    parser.add_argument("--initial-prompt", type=str, default=get_env_str("INITIAL_PROMPT", None),
                        help="Initial prompt to use for transcription")
    parser.add_argument("--temperature", type=float, default=get_env_float("TEMPERATURE", 0.0),
                        help="Temperature to use for transcription")
    parser.add_argument("--vtt-max-chars", type=int, default=get_env_int("VTT_MAX_CHARS", 0),
                        help="Maximum characters per VTT segment (0 for no limit)")
    return parser.parse_args()


def get_hf_token():
    """Check env, if in colab use colab env, otherwise use sys env for HF token"""
    if is_running_in_colab():
        token = HfFolder().get_token()
        if not token:
            raise ValueError(
                "Hugging Face token not found. Please run `notebook_login()` in Colab."
            )
        return token
    else:
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

def transcribe_audio(audio_path: str, model_size: str, task: str, language: Optional[str], transcription_model: str, cpu_threads: int, beam_size: int, vad_level: int, num_workers: int, precision: str, gpu_device: int, initial_prompt: Optional[str], temperature: float, num_speakers: Optional[int] = 1) -> Iterator[Dict[str, any]]:
    """
    Transcribe audio using Whisper and return a generator of segments.
    This function is memory-efficient, yielding segments as they are transcribed.
    """
    if not os.path.exists(audio_path):
        raise RuntimeError(f"Audio file not found: {audio_path}")

    print(f"Loading Whisper model: {model_size}")

    # --- Determine Device and Compute Type ---
    precision_hint = precision.lower()
    device = "cpu"
    compute_type = "int8" # Default fallback

    if torch.cuda.is_available():
        print(f"CUDA GPU detected. Trying to use GPU device {gpu_device}.")
        device = f"cuda:{gpu_device}"
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
    beam_size_int = beam_size
    print(f"Using beam size: {beam_size_int}")
    # --- End Determine Beam Size ---

    whisper_cache_dir = "/app/models/whisper"
    
    vad_filter = True
    if vad_level == 0:
        vad_filter = False
        vad_params = {}
    elif vad_level == 1:
        vad_params = {'min_silence_duration_ms': 1000}
    elif vad_level == 2:
        vad_params = {'min_silence_duration_ms': 500}
    elif vad_level == 3:
        vad_params = {'min_silence_duration_ms': 250}
    else:
        vad_params = {'min_silence_duration_ms': 500}


    if transcription_model == "faster-whisper":
        # --- Faster-Whisper Implementation ---
        try:
            model_repo_map = {
                "large-v2": "large-v2", "large-v3": "large-v3", "large": "large-v2",
                "medium": "medium", "small": "small", "base": "base", "tiny": "tiny",
                "medium.en": "medium.en", "small.en": "small.en", "base.en": "base.en", "tiny.en": "tiny.en",
            }
            faster_model_size = model_repo_map.get(model_size, model_size)
            is_distilled = "distil" in faster_model_size
            if is_distilled and not faster_model_size.startswith("Systran/faster-"):
                model_id = f"Systran/faster-{faster_model_size}"
            else:
                model_id = f"guillaumekln/faster-whisper-{faster_model_size}"

            print(f"Attempting to load faster-whisper model: {model_id} with device: {device}, compute_type: {compute_type}")

            total_cpu_threads = psutil.cpu_count(logical=True)
            physical_cores = psutil.cpu_count(logical=False)
            
            if device == "cpu":
                if cpu_threads > 0:
                    optimal_threads = cpu_threads
                else:
                    cpu_threshold = get_env_float("CPU_THRESHOLD", 0.75)
                    optimal_threads = max(1, int(total_cpu_threads * cpu_threshold))
                
                if num_workers > 0:
                    optimal_workers = num_workers
                else:
                    min_workers = get_env_int("MIN_WORKERS", 2)
                    optimal_workers = min(4, max(min_workers, physical_cores // 4))
                print(f"CPU Mode: Using {optimal_threads} threads and {optimal_workers} workers.")
            else:
                optimal_threads = physical_cores
                if num_workers > 0:
                    optimal_workers = num_workers
                else:
                    min_workers = get_env_int("MIN_WORKERS", 2)
                    optimal_workers = min_workers
                print(f"GPU Mode: Using {optimal_threads} threads for CPU ops and {optimal_workers} workers.")

            model = WhisperModel(
                model_id, device=device, compute_type=compute_type,
                download_root=whisper_cache_dir, cpu_threads=optimal_threads, num_workers=optimal_workers
            )

            print(f"Transcribing {audio_path} using faster-whisper...")
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*MPEG_LAYER_III.*")
                segments_generator, info = model.transcribe(
                    audio_path, task=task, language=language,
                    vad_filter=vad_filter, vad_parameters=vad_params, beam_size=beam_size_int,
                    initial_prompt=initial_prompt, temperature=temperature
                )
            print(f"Detected language: {info.language} with probability {info.language_probability:.2f}\n")

            # Yield segments directly from the generator
            segment_count = 0
            total_segments = 0 # We don't know the total, but we can count as we go
            
            print("Transcription in progress...", flush=True)
            for segment in segments_generator:
                segment_count += 1
                print(f"  - Transcribed segment {segment_count}: \"{segment.text[:50]}...\"", flush=True)
                yield {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text
                }
            print(f"Transcription complete. Found {segment_count} segments.", flush=True)
            return

        except ImportError:
            print("faster-whisper package not found or import failed.")
            raise
        except Exception as e:
            print(f"Error loading or using faster-whisper model '{model_id}': {str(e)}")
            raise

    elif transcription_model == "openai-whisper":
        # --- Fallback to OpenAI's whisper package ---
        try:
            import whisper
            print("Using OpenAI's whisper package.")
            cache_path = Path(whisper_cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)

            model = whisper.load_model(
                model_size, download_root=str(cache_path), in_memory=False
            )

            print(f"Transcribing {audio_path} with OpenAI's whisper...")
            result = model.transcribe(audio_path, task=task, language=language, beam_size=beam_size_int, initial_prompt=initial_prompt, temperature=temperature)

            # Yield segments from the result
            for segment in result["segments"]:
                yield {
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"]
                }
        except ImportError:
            raise RuntimeError("OpenAI whisper could not be used.")
        except Exception as e:
            raise RuntimeError(f"Could not transcribe audio using OpenAI whisper: {str(e)}")
    else:
        raise ValueError(f"Unknown transcription model: {transcription_model}")


def perform_diarization(audio_path: str, diarization_pipeline: str, num_speakers: Optional[int] = None) -> Optional[dict]:
    """Perform speaker diarization using pyannote.audio"""
    # Skip diarization for single speaker
    if num_speakers == 1:
        print("Skipping diarization for single speaker audio")
        return None
        
    print(f"Loading diarization pipeline: {diarization_pipeline}")
    
    # Set a timeout for the diarization process
    diarization_timeout = get_env_int("DIARIZATION_TIMEOUT", 1800) # 30 minutes default
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(diarization_timeout)
    
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
            
        try:
            print(f"Attempting to load pipeline from {diarization_pipeline}...")

            # Specific pre-flight check for pyannote/speaker-diarization-3.1's critical dependency
            if diarization_pipeline == "pyannote/speaker-diarization-3.1":
                segmentation_dependency_model = "pyannote/segmentation-3.0"
                try:
                    print(f"INFO: Performing pre-flight check for essential dependency: {segmentation_dependency_model}...")
                    # First check if the file is already cached to avoid redundant downloads
                    from huggingface_hub import hf_hub_download
                    
                    try:
                        # Try to access the cached file first
                        cached_file = hf_hub_download(
                            repo_id=segmentation_dependency_model,
                            filename="config.yaml",
                            use_auth_token=token,
                            local_files_only=True  # Only check cache, don't download
                        )
                        print(f"INFO: Pre-flight check for {segmentation_dependency_model} successful. Using cached file: {cached_file}")
                    except Exception:
                        # File not in cache, download it
                        print(f"INFO: Config not cached, downloading for first time...")
                        downloaded_file = hf_hub_download(
                            repo_id=segmentation_dependency_model,
                            filename="config.yaml",
                            use_auth_token=token
                        )
                        print(f"INFO: Pre-flight check for {segmentation_dependency_model} successful. Downloaded to cache: {downloaded_file}")
                except Exception as dep_check_e:
                    # This exception is raised if hf_hub_download fails for the dependency.
                    print(f"\nCRITICAL ERROR: Failed to access or download the critical dependency '{segmentation_dependency_model}'.")
                    print(f"  This model is required by the preferred diarization pipeline '{diarization_pipeline}'.")
                    print(f"  Error details from dependency check: {str(dep_check_e)}")
                    print("\n  Common reasons for this failure include:")
                    print(f"  1. Missing user agreement: You might need to visit https://huggingface.co/{segmentation_dependency_model} and accept its terms of use.")
                    print(f"  2. Invalid or missing HF_TOKEN: Ensure your Hugging Face token is correctly set in the .env file and has the necessary permissions.")
                    print(f"  3. Network issues or model availability problems on Hugging Face Hub.")
                    print("\nThe script cannot continue without this component and will now exit.")
                    sys.exit(1) # Exit the script immediately

            pipeline = Pipeline.from_pretrained(
                diarization_pipeline,
                use_auth_token=token
            )
            
            if pipeline is not None:
                print(f"✓ Successfully loaded {diarization_pipeline}")
            else:
                raise RuntimeError(f"Failed to load diarization pipeline: {diarization_pipeline}")
                
        except Exception as e:
            print(f"✗ Failed to load {diarization_pipeline}: {str(e)}")
            raise e
                
        print("Pipeline loaded successfully, setting device...")
        # Set to CPU - we'll enhance this later to handle GPU if available
        pipeline.to(torch.device("cpu"))
        
        print("Performing speaker diarization...")
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*MPEG_LAYER_III.*")
            if num_speakers is not None:
                print(f"Using specified number of speakers: {num_speakers}")
                diarization = pipeline(audio_path, num_speakers=num_speakers)
            else:
                print("Automatically determining number of speakers")
                diarization = pipeline(audio_path)
        
        return diarization

    
    except TimeoutError:
        print(f"\nERROR: Diarization timed out after {diarization_timeout} seconds.")
        print("This can happen with very long audio files or complex speaker patterns.")
        print("You can try increasing the timeout by setting the DIARIZATION_TIMEOUT environment variable.")
        return None
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
    finally:
        # Disable the alarm
        signal.alarm(0)

def combine_transcription_with_diarization(segments_generator: Iterator[Dict[str, any]], diarization) -> Iterator[Dict[str, any]]:
    """
    Combine a stream of transcription segments with speaker information using an efficient
    linear-time algorithm. This is a generator function that yields enhanced segments one by one.
    """
    print("Combining transcription with speaker diarization (efficient algorithm)...")
    
    # Prepare speaker turns for efficient access
    # pyannote's itertracks is already sorted by time, which is crucial.
    diarization_turns = list(diarization.itertracks(yield_label=True))
    turn_idx = 0

    # Use a log-friendly tqdm configuration
    progress_bar = tqdm(
        segments_generator, 
        desc="Combining segments", 
        unit="segment",
        file=sys.stdout,  # Force output to stdout
        ascii=True,       # Use ASCII characters for the bar
        mininterval=10    # Update every 10 seconds
    )

    for segment in progress_bar:
        segment_start = segment["start"]
        segment_end = segment["end"]
        
        best_speaker = None
        max_overlap = 0

        # Advance through speaker turns that are completely before the current segment
        while turn_idx < len(diarization_turns) and diarization_turns[turn_idx][0].end < segment_start:
            turn_idx += 1

        # Now, check for overlapping turns starting from the current position
        temp_idx = turn_idx
        while temp_idx < len(diarization_turns):
            turn, _, speaker = diarization_turns[temp_idx]

            # If the turn starts after the segment ends, no more relevant turns for this segment
            if turn.start > segment_end:
                break

            # Calculate overlap
            overlap_start = max(segment_start, turn.start)
            overlap_end = min(segment_end, turn.end)
            overlap_duration = overlap_end - overlap_start
            
            if overlap_duration > max_overlap:
                max_overlap = overlap_duration
                best_speaker = speaker
            
            temp_idx += 1

        segment["speaker"] = best_speaker if best_speaker else DEFAULT_SPEAK_LABEL
        yield segment

def split_segment_by_chars(segment: Dict[str, any], max_chars: int) -> List[Dict[str, any]]:
    """Split a segment into smaller segments based on character count."""
    if max_chars <= 0 or len(segment["text"]) <= max_chars:
        return [segment]
    
    text = segment["text"]
    start_time = segment["start"]
    end_time = segment["end"]
    duration = end_time - start_time
    speaker = segment.get("speaker", DEFAULT_SPEAK_LABEL)
    
    segments = []
    words = text.split()
    current_text = ""
    current_words = []
    
    for word in words:
        # Check if adding this word would exceed the limit
        test_text = current_text + (" " if current_text else "") + word
        if len(test_text) > max_chars and current_text:
            # Calculate timing for this sub-segment
            chars_processed = len(" ".join(current_words))
            char_ratio = chars_processed / len(text)
            segment_start = start_time + (duration * (len(" ".join(segments)) / len(text)) if segments else 0)
            segment_end = start_time + (duration * char_ratio)
            
            segments.append({
                "start": segment_start,
                "end": segment_end,
                "text": current_text.strip(),
                "speaker": speaker
            })
            
            current_text = word
            current_words = [word]
        else:
            if current_text:
                current_text += " " + word
            else:
                current_text = word
            current_words.append(word)
    
    # Add the remaining text as the final segment
    if current_text:
        chars_used = sum(len(s["text"]) + 1 for s in segments) - 1 if segments else 0
        char_ratio = chars_used / len(text) if len(text) > 0 else 0
        segment_start = start_time + (duration * char_ratio)
        
        segments.append({
            "start": segment_start,
            "end": end_time,
            "text": current_text.strip(),
            "speaker": speaker
        })
    
    return segments

def save_as_vtt(segments_generator: Iterator[Dict[str, any]], output_path: str, max_chars: int = 0):
    """Save the transcription as a WebVTT file from a generator."""
    vtt = webvtt.WebVTT()
    
    print("Saving WebVTT file...")
    if max_chars > 0:
        print(f"Splitting segments to maximum {max_chars} characters each")
    
    # Use a log-friendly tqdm configuration
    progress_bar = tqdm(
        segments_generator, 
        desc="Saving VTT", 
        unit="caption",
        file=sys.stdout,
        ascii=True,
        mininterval=10
    )

    for segment in progress_bar:
        # Split segment if character limit is specified
        if max_chars > 0:
            sub_segments = split_segment_by_chars(segment, max_chars)
        else:
            sub_segments = [segment]
        
        # Create VTT captions for each sub-segment
        for sub_segment in sub_segments:
            caption = webvtt.Caption(
                start=format_timestamp(sub_segment["start"]),
                end=format_timestamp(sub_segment["end"]),
                text=f"<v {sub_segment.get('speaker', DEFAULT_SPEAK_LABEL)}>{sub_segment['text']}</v>"
            )
            vtt.captions.append(caption)
    
    vtt.save(output_path)
    print(f"Saved WebVTT file to {output_path}")

def save_as_json(segments_generator: Iterator[Dict[str, any]], output_path: str):
    """Save the transcription as a JSON file from a generator."""
    print("Saving JSON file...")
    segments_list = list(segments_generator)
    with open(output_path, "w") as f:
        import json
        json.dump(segments_list, f, indent=2)
    print(f"Saved JSON file to {output_path}")

def save_as_srt(segments_generator: Iterator[Dict[str, any]], output_path: str):
    """Save the transcription as a SRT file from a generator."""
    print("Saving SRT file...")
    with open(output_path, "w") as f:
        for i, segment in enumerate(segments_generator):
            f.write(f"{i+1}\n")
            f.write(f"{format_timestamp_srt(segment['start'])} --> {format_timestamp_srt(segment['end'])}\n")
            f.write(f"{segment['text']}\n\n")
    print(f"Saved SRT file to {output_path}")

def format_timestamp(seconds: float) -> str:
    """Format timestamp for WebVTT format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

def format_timestamp_srt(seconds: float) -> str:
    """Format timestamp for SRT format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds * 1000) % 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

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
    if is_running_in_colab():
        benchmark_file = "/content/data/whisper_benchmarks.csv"
    else:
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
        model, processing_time, (processing_time/duration if duration > 0 else 0),
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
    if duration > 0:
        print(f"Real-time Factor: {(processing_time/duration):.2f}x")
    print(f"CPU Usage: {cpu_percent}% (of {cpu_count} CPUs)")
    print(f"Memory Usage: {memory_percent}%")
    print(f"Results saved to: {benchmark_file}")

def estimate_processing_time(duration: float, model: str, num_speakers: Optional[int] = None) -> dict:
    """Estimate processing time based on file duration, model size, and speakers"""
    # Handle zero duration gracefully
    if duration <= 0:
        print("WARNING: Audio duration is zero or negative. Cannot estimate processing time.")
        return {
            "total_minutes": 0,
            "transcription_minutes": 0,
            "diarization_minutes": 0,
            "real_time_factor": 0
        }
        
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
        "real_time_factor": total_time / duration if duration > 0 else 0
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
    if estimate['real_time_factor'] > 0:
        print(f"Estimated real-time factor: {estimate['real_time_factor']:.1f}x")
    print("Note: Estimates are approximate and may vary based on CPU speed and audio complexity\n")

def main():
    args = setup_args()
    
    # Verify input file exists
    if not os.path.exists(args.audio_file):
        print(f"ERROR: Input file not found: {args.audio_file}")
        sys.exit(1)
    
    # Verify it's actually a file and not a directory
    if os.path.exists(args.audio_file) and os.path.isdir(args.audio_file):
        print(f"ERROR: {args.audio_file} exists but is not a regular file (might be a directory)")
        print("This is often caused by Docker volume mount issues on macOS.")
        print("Try using a different audio file or check Docker file sharing permissions.")
        sys.exit(1)
    
    # Get audio duration
    duration = get_audio_duration(args.audio_file)
    print(f"Audio duration: {duration:.2f} seconds")
    
    # Check if we have a valid duration
    if duration <= 0:
        print("ERROR: Could not determine audio duration or audio file is empty/corrupt.")
        print("Please check that the file is a valid audio file.")
        sys.exit(1)
    
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

    # check HF token
    check_hf_token()
    
    # --- Start of Timed Processing ---
    overall_start_time = time.time()
    
    # Validate the output path if provided
    output_path = args.output
    if not output_path:
        # Generate default output path by replacing extension
        input_basename = os.path.splitext(os.path.basename(args.audio_file))[0]
        output_path = f"/output/{input_basename}.{args.format}"
        print(f"No output path specified, using: {output_path}")
    
    # Verify input file exists
    if not os.path.exists(args.audio_file):
        print(f"ERROR: Input file not found: {args.audio_file}")
        sys.exit(1)


    # test if mp4 can be diarized, if not then convert to wav form and continue processing
    if args.audio_file.lower().endswith(".mp4"):
        if not can_load_mp4_audio(args.audio_file):
            args.audio_file = convert_mp4_to_wav(args.audio_file)
            print(f"Unable to diarize MP4, converted to WAV file: {args.audio_file}")
        else:
            print(f"Can diarize MP4 using original file: {args.audio_file}")

    try:
        # --- Start of Streaming Pipeline ---
        
        # 1. Transcribe audio to get a generator of segments
        transcription_start_time = time.time()
        segments_generator = transcribe_audio(
            args.audio_file, args.model, args.task, args.language, 
            args.transcription_model, args.cpu_threads, args.beam_size, 
            args.vad_level, args.num_workers, args.precision, args.gpu_device, 
            args.initial_prompt, args.temperature, args.num_speakers
        )
        # The generator is lazy, so we wrap it to time the actual transcription process
        def timed_transcription_generator(gen):
            yield from gen
            transcription_time = time.time() - transcription_start_time
            print(f"DEBUG: Transcription finished in {transcription_time:.2f} seconds.")
        
        segments_generator = timed_transcription_generator(segments_generator)

        # 2. Perform diarization (if needed)
        diarization_start_time = time.time()
        if args.num_speakers != 1:
            diarization = perform_diarization(
                args.audio_file, 
                diarization_pipeline=args.diarization_pipeline, 
                num_speakers=args.num_speakers
            )
            diarization_time = time.time() - diarization_start_time
            print(f"DEBUG: Diarization finished in {diarization_time:.2f} seconds.")
            
            if diarization:
                # Combine transcription with speaker info in a streaming fashion
                combination_start_time = time.time()
                enhanced_segments_generator = combine_transcription_with_diarization(segments_generator, diarization)
                
                def timed_combination_generator(gen):
                    yield from gen
                    combination_time = time.time() - combination_start_time
                    print(f"DEBUG: Combination finished in {combination_time:.2f} seconds.")
                
                enhanced_segments_generator = timed_combination_generator(enhanced_segments_generator)

            else:
                print("WARNING: Diarization failed. Proceeding with transcription only.")
                enhanced_segments_generator = (dict(segment, speaker=DEFAULT_SPEAK_LABEL) for segment in segments_generator)
        else:
            print("Single speaker mode: Skipping diarization.")
            enhanced_segments_generator = (dict(segment, speaker="SPEAKER_01") for segment in segments_generator)

        # 3. Save the results from the generator
        if args.format == "vtt":
            save_as_vtt(enhanced_segments_generator, output_path, args.vtt_max_chars)
        elif args.format == "srt":
            save_as_srt(enhanced_segments_generator, output_path)
        elif args.format == "txt":
            with open(output_path, "w") as f:
                for segment in enhanced_segments_generator:
                    f.write(f"[{segment.get('speaker', DEFAULT_SPEAK_LABEL)}] {segment['text']}\n")
            print(f"Saved plain text file to {output_path}")
        elif args.format == "json":
            save_as_json(enhanced_segments_generator, output_path)
        
        # --- End of Streaming Pipeline ---

        # Calculate total processing time and log benchmark
        processing_time = time.time() - overall_start_time
        log_benchmark(
            audio_path=args.audio_file,
            duration=duration,
            num_speakers=args.num_speakers,
            model=args.model,
            processing_time=processing_time
        )
        
        print(f"Processing complete! Output saved to {output_path}")

    except Exception as e:
        print(f"ERROR: Processing failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()