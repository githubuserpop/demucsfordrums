import torch
import torchaudio
import os
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import numpy as np

class DrumDataset(Dataset):
    def __init__(self, 
                 sample_dirs,      # directories with individual drum samples
                 stem_dirs,        # directories with full song stems
                 sources=["kicks", "snares", "claps", "hi hats", "open hats", "percs"],
                 sample_rate=44100,
                 segment_duration=4.0,
                 channels=2,
                 normalize=True,
                 sample_ratio=0.3  # ratio of individual samples vs full songs
                ):
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.segment_length = int(segment_duration * sample_rate)
        self.channels = channels
        self.normalize = normalize
        self.sources = sources
        self.sample_ratio = sample_ratio
        
        # Load paths for individual samples
        self.sample_paths = {source: [] for source in sources}
        for dir_path in sample_dirs:
            for source in sources:
                source_path = Path(dir_path) / source
                if source_path.exists():
                    self.sample_paths[source].extend(
                        [f for f in source_path.glob("*.wav") or source_path.glob("*.mp3")]
                    )
        
        # Load paths for full stems
        self.stem_paths = []
        self.individual_stem_paths = []  # For stems that have individual drum tracks
        
        for dir_path in stem_dirs:
            dir_path = Path(dir_path)
            # Check if this directory has individual drum stems
            has_individual_stems = any((dir_path / source).exists() for source in sources)
            
            if has_individual_stems:
                # This is a directory with individual drum stems (like beat stems)
                for source in sources:
                    source_path = dir_path / source
                    if source_path.exists():
                        self.individual_stem_paths.extend(
                            [(f, source) for f in source_path.glob("*.wav")]
                        )
            else:
                # This is a directory with only full drum stems (like song drum stems)
                if (dir_path / "drums").exists():
                    self.stem_paths.extend(
                        [(f, "drums") for f in (dir_path / "drums").glob("*.wav")]
                    )
    
    def __len__(self):
        return (len(self.stem_paths) + 
                len(self.individual_stem_paths) + 
                sum(len(paths) for paths in self.sample_paths.values()))
    
    def _load_and_process_audio(self, path, target_length):
        audio, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            audio = torchaudio.transforms.Resample(sr, self.sample_rate)(audio)
        
        # Convert to stereo if needed
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        
        # Handle length
        if audio.shape[1] < target_length:
            # Pad if too short
            padding = target_length - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, padding))
        elif audio.shape[1] > target_length:
            # Random crop if too long
            start = random.randint(0, audio.shape[1] - target_length)
            audio = audio[:, start:start + target_length]
            
        return audio
    
    def __getitem__(self, idx):
        # First decide based on sample_ratio whether to use individual samples or stems
        use_sample = random.random() < self.sample_ratio
        
        if use_sample:
            # Use individual samples from organized drums folder
            source = random.choice(self.sources)
            sample_path = random.choice(self.sample_paths[source])
            
            # Initialize tensors
            mix = torch.zeros(self.channels, self.segment_length)
            targets = {s: torch.zeros(self.channels, self.segment_length) for s in self.sources}
            
            # Load the chosen sample
            audio = self._load_and_process_audio(sample_path, self.segment_length)
            targets[source] = audio
            mix = audio
        
        else:
            # Handle stems (both individual and full)
            total_stems = len(self.individual_stem_paths) + len(self.stem_paths)
            if total_stems == 0:
                # Fallback to samples if no stems available
                return self.__getitem__(0)
                
            # Decide whether to use individual stem or full stem
            use_individual = (len(self.individual_stem_paths) > 0 and 
                            (len(self.stem_paths) == 0 or random.random() < 0.7))  # Favor individual stems
            
            if use_individual:
                # Use individual stems from beat stems
                idx = random.randint(0, len(self.individual_stem_paths) - 1)
                stem_path, source_type = self.individual_stem_paths[idx]
                
                # Initialize tensors
                mix = torch.zeros(self.channels, self.segment_length)
                targets = {s: torch.zeros(self.channels, self.segment_length) for s in self.sources}
                
                # Load the audio for this stem
                audio = self._load_and_process_audio(stem_path, self.segment_length)
                targets[source_type] = audio
                mix = audio
            
            else:
                # Use full drum stems from song drum stems
                idx = random.randint(0, len(self.stem_paths) - 1)
                stem_path, _ = self.stem_paths[idx]
                
                # Load the full drum stem
                mix = self._load_and_process_audio(stem_path, self.segment_length)
                
                # For full stems, initialize empty targets
                targets = {s: torch.zeros(self.channels, self.segment_length) for s in self.sources}
        
        # Normalize if required
        if self.normalize:
            mix_std = mix.std()
            if mix_std > 0:
                mix = mix / mix_std
                for source in self.sources:
                    targets[source] = targets[source] / mix_std
        
        return mix, targets

def get_drum_datasets(args, split='train'):
    """Create train and valid datasets for drum separation."""
    # Get dataset paths from config
    sample_dirs = args.dset.sample_dirs
    stem_dirs = args.dset.stem_dirs
    
    if not sample_dirs or not stem_dirs:
        raise ValueError("No dataset paths provided in config. Please add 'sample_dirs' and 'stem_dirs' to dset config.")
    
    # Verify paths exist
    sample_dirs = [d for d in sample_dirs if os.path.exists(d)]
    stem_dirs = [d for d in stem_dirs if os.path.exists(d)]
    
    if not sample_dirs or not stem_dirs:
        raise ValueError("No valid dataset paths found. Please check the paths in config.")
    
    # Split directories for training and validation
    train_sample_dirs = sample_dirs[:-1]
    train_stem_dirs = stem_dirs[:-1]
    valid_sample_dirs = sample_dirs[-1:]
    valid_stem_dirs = stem_dirs[-1:]
    
    if split == 'train':
        dataset = DrumDataset(
            sample_dirs=train_sample_dirs,
            stem_dirs=train_stem_dirs,
            sources=args.dset.sources,
            sample_rate=args.dset.samplerate,
            segment_duration=args.dset.segment,
            channels=args.dset.channels,
            normalize=args.dset.normalize,
            sample_ratio=args.dset.sample_ratio
        )
    else:
        dataset = DrumDataset(
            sample_dirs=valid_sample_dirs,
            stem_dirs=valid_stem_dirs,
            sources=args.dset.sources,
            sample_rate=args.dset.samplerate,
            segment_duration=args.dset.segment,
            channels=args.dset.channels,
            normalize=args.dset.normalize,
            sample_ratio=args.dset.sample_ratio
        )
    
    return DataLoader(
        dataset,
        batch_size=args.dset.batch_size,
        shuffle=(split == 'train'),
        num_workers=args.misc.num_workers,
        pin_memory=True
    )