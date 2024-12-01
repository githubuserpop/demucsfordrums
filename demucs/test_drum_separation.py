import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchaudio
import soundfile as sf
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from demucs.hdemucs import HDemucs

def load_audio(path, device="cpu", chunk_size=None):
    """Load audio file and convert to model input format with optional chunking."""
    try:
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 2:
            wav = wav[:2]
        elif wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
            
        if chunk_size is not None:
            # Process in chunks to save memory
            chunks = wav.split(chunk_size, dim=1)
            return chunks, sr
        return wav.to(device), sr
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None, None

def evaluate_separation(model, mix_path, output_dir, device="cpu", chunk_size=262144):
    """Evaluate drum separation on a single mix with memory-efficient processing."""
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load and process audio in chunks
        chunks, sr = load_audio(mix_path, device, chunk_size)
        if chunks is None:
            return None
            
        components = ["kick", "clap", "snare", "hihat", "openhat", "perc"]
        separated_chunks = {comp: [] for comp in components}
        
        print("Processing audio chunks...")
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}")
            chunk = chunk.to(device)
            
            with torch.no_grad():
                separated = model(chunk[None])[0]
                
            # Move results to CPU and clear GPU memory
            for idx, comp in enumerate(components):
                separated_chunks[comp].append(separated[idx].cpu())
            
            # Clear GPU cache
            if device == "cuda":
                torch.cuda.empty_cache()
        
        print("Saving separated stems...")
        for comp in components:
            # Concatenate chunks
            stem = torch.cat(separated_chunks[comp], dim=1)
            save_path = output_dir / f"{comp}.wav"
            torchaudio.save(save_path, stem, sr)
            
        return {
            "components": components,
            "sample_rate": sr,
            "length": sum(chunk.shape[-1] for chunk in chunks)
        }
        
    except Exception as e:
        print(f"Error in separation: {e}")
        return None

def create_test_mix(max_duration=30):
    """Create a shorter test mix for testing purposes."""
    try:
        base_dir = Path("test_data/organized_drums")  # Use a local test directory
        output_dir = Path("test_data/output")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        components = {
            "kicks": 1,
            "snares": 1,
            "hi hats": 1,  # Reduced from 2
            "claps": 1,
            "open hats": 1,
            "percs": 1
        }
        
        mix = None
        sr = None
        max_samples = max_duration * 44100  # Limit duration
        
        for component, count in components.items():
            component_dir = base_dir / component
            if not component_dir.exists():
                print(f"Warning: {component_dir} not found, skipping...")
                continue
                
            samples = list(component_dir.glob("*.wav"))[:count]
            
            for sample_path in samples:
                wav, sample_sr = torchaudio.load(str(sample_path))
                if sr is None:
                    sr = sample_sr
                
                # Ensure consistent sample rate
                if sample_sr != sr:
                    wav = torchaudio.transforms.Resample(sample_sr, sr)(wav)
                
                # Limit duration
                if wav.shape[1] > max_samples:
                    wav = wav[:, :max_samples]
                
                if mix is None:
                    mix = wav
                else:
                    # Pad shorter audio to match lengths
                    if wav.shape[1] < mix.shape[1]:
                        wav = torch.nn.functional.pad(wav, (0, mix.shape[1] - wav.shape[1]))
                    elif mix.shape[1] < wav.shape[1]:
                        mix = torch.nn.functional.pad(mix, (0, wav.shape[1] - mix.shape[1]))
                    mix = mix + wav
        
        if mix is None:
            raise ValueError("No audio files were loaded")
            
        # Normalize mix
        mix = mix / mix.abs().max()
        
        # Save mix
        output_path = output_dir / "test_mix.wav"
        torchaudio.save(output_path, mix, sr)
        return str(output_path)
        
    except Exception as e:
        print(f"Error creating test mix: {e}")
        return None

def main():
    """Main testing function with optimized model parameters."""
    try:
        # Initialize model with reduced complexity
        model = HDemucs(
            channels=32,  # Reduced from 48
            growth=1.5,   # Reduced from 2
            depth=4,      # Reduced from 6
            freq_emb_dim=16  # Reduced from 32
        ).to("cpu")  # Start on CPU
        
        # Create test mix
        mix_path = create_test_mix(max_duration=30)  # Limit test duration
        if not mix_path:
            print("Failed to create test mix")
            return
            
        # Evaluate separation
        output_dir = Path("test_data/separated")
        results = evaluate_separation(model, mix_path, output_dir, device="cpu")
        
        if results:
            print("\nSeparation completed successfully!")
            print(f"Components: {results['components']}")
            print(f"Sample rate: {results['sample_rate']} Hz")
            print(f"Audio length: {results['length']} samples")
        else:
            print("Separation failed")
            
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()