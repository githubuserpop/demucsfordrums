"""Data loading utilities for Demucs."""
import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

def load_custom_data(training_folders, batch_size):
    """
    Function to set up the dataloader for training data across the specified folders
    with variable sample rates.
    """
    class DrumDataset(Dataset):
        def __init__(self, folders):
            self.audio_files = []
            for folder in folders:
                if os.path.exists(folder):
                    for root, _, files in os.walk(folder):
                        for file in files:
                            if file.endswith(('.wav', '.mp3')):
                                self.audio_files.append(os.path.join(root, file))

        def __len__(self):
            return len(self.audio_files)

        def __getitem__(self, idx):
            audio_path = self.audio_files[idx]
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Ensure stereo
            if waveform.size(0) == 1:
                waveform = waveform.repeat(2, 1)
            
            # Normalize
            waveform = waveform / waveform.abs().max()
            
            return waveform

    dataset = DrumDataset(training_folders)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
