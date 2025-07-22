import sys
import torch
import torchaudio
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from speechbrain.pretrained import EncoderClassifier

def count_speakers_fast(audio_path):
    try:
        # Load speaker embedding model
        spk_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )
        
        # Process audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        
        # Handle mono/stereo - ensure we're working with mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Split into chunks
        chunk_size = 16000 * 3  # 3 seconds at 16kHz
        chunks = torch.split(waveform[0], chunk_size)
        
        # Extract embeddings
        embeddings = []
        for chunk in enumerate(chunks):
            if len(chunk) > 8000:  # Minimum 0.5 second
                emb = spk_model.encode_batch(chunk.unsqueeze(0))
                embeddings.append(emb.squeeze().cpu().numpy())
        
        # Not enough embeddings for clustering
        if len(embeddings) <= 1:
            return 1
            
        # Cluster embeddings
        if len(embeddings) > 1:
            embeddings_array = np.vstack(embeddings)
            
            # Fix: Remove affinity parameter
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.5,
                linkage='average'
            ).fit(embeddings_array)
            
            num_speakers = len(set(clustering.labels_))
            return max(1, num_speakers)  # At least 1 speaker
        
        return 1
        
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return 0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(0)
        sys.exit(1)
    print("Counting the number of speakers.")
    audio_file = sys.argv[1]
    print(count_speakers_fast(audio_file))