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

def load_audio(path, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Load audio file and convert to model input format."""
    try:
        # Try loading with torchaudio first
        wav, sr = torchaudio.load(path)
    except RuntimeError:
        # Fallback to soundfile
        wav, sr = sf.read(str(path))
        wav = torch.from_numpy(wav.T)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
    
    if wav.shape[0] > 2:
        wav = wav[:2]
    elif wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    wav = wav.to(device)
    return wav, sr

def evaluate_separation(model, mix_path, output_dir, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Evaluate drum separation on a single mix."""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load and process audio
    wav, sr = load_audio(mix_path, device)
    ref_length = wav.shape[-1]
    
    # Apply separation
    with torch.no_grad():
        separated = model(wav[None])[0]
    
    # Get component names
    components = ["kick", "clap", "snare", "hihat", "openhat", "perc"]
    
    # Save separated stems
    for idx, component in enumerate(components):
        stem = separated[idx]
        stem = stem.cpu()
        save_path = output_dir / f"{component}.wav"
        try:
            torchaudio.save(save_path, stem, sr)
        except RuntimeError:
            # Fallback to soundfile for saving
            sf.write(str(save_path), stem.numpy().T, sr)
        
        # Plot spectrogram
        plt.figure(figsize=(10, 4))
        spec = torchaudio.transforms.Spectrogram()(stem[0])
        plt.imshow(spec.log2()[:-1, :].numpy(), aspect='auto', origin='lower')
        plt.title(f'{component} Spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.savefig(output_dir / f"{component}_spec.png")
        plt.close()
    
    return {
        "components": components,
        "sample_rate": sr,
        "length": ref_length
    }

def create_test_mix():
    """Create a test mix from organized drum samples."""
    base_dir = Path("/DATA/Training Data/organized drums")
    output_dir = Path("/DATA/demucs/test_data")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Component directories and sample counts
    components = {
        "kicks": 1,
        "snares": 1,
        "hi hats": 2,
        "claps": 1,
        "open hats": 1,
        "percs": 1
    }
    
    mix = None
    sr = None
    
    # Load and mix samples
    for component, count in components.items():
        print(f"Processing {component}...")
        component_dir = base_dir / component
        samples = list(component_dir.glob("*.wav"))[:count]
        
        for sample_path in samples:
            print(f"  Loading {sample_path.name}")
            try:
                wav, sample_sr = load_audio(str(sample_path))
                if sr is None:
                    sr = sample_sr
                
                # Initialize or add to mix
                if mix is None:
                    mix = wav
                else:
                    # Pad shorter sample if needed
                    if wav.shape[1] < mix.shape[1]:
                        wav = torch.nn.functional.pad(wav, (0, mix.shape[1] - wav.shape[1]))
                    elif wav.shape[1] > mix.shape[1]:
                        mix = torch.nn.functional.pad(mix, (0, wav.shape[1] - mix.shape[1]))
                    mix = mix + wav
            except Exception as e:
                print(f"  Warning: Could not load {sample_path.name}: {str(e)}")
                continue
    
    if mix is None:
        raise RuntimeError("Failed to create mix: no valid audio files found")
    
    # Normalize mix
    mix = mix / mix.abs().max()
    
    # Save mix
    mix_path = output_dir / "test_mix.wav"
    try:
        torchaudio.save(str(mix_path), mix, sr)
    except RuntimeError:
        # Fallback to soundfile for saving
        sf.write(str(mix_path), mix.numpy().T, sr)
    
    return str(mix_path)

def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create test mix
    print("Creating test mix...")
    test_mix_path = create_test_mix()
    print(f"Test mix created at: {test_mix_path}")
    
    # Initialize model
    print("Initializing model...")
    model = HDemucs(
        channels=48,
        growth=2,
        nfft=4096,
        depth=6,
        freq_emb_dim=32,
        normalize=True
    ).to(device)
    
    # Test directories
    output_dir = "/DATA/demucs/test_results"
    
    # Run evaluation
    print("Starting separation...")
    results = evaluate_separation(model, test_mix_path, output_dir, device)
    
    print("\nSeparation completed!")
    print(f"Components separated: {results['components']}")
    print(f"Sample rate: {results['sample_rate']} Hz")
    print(f"Audio length: {results['length']} samples")
    print(f"\nResults saved to: {output_dir}")
    print("Generated files:")
    print("- Separated audio (.wav) for each component")
    print("- Spectrograms (.png) for visual analysis")

if __name__ == "__main__":
    main()