#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Main training script entry point"""

import logging
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torchaudio

from dora import hydra_main, get_xp
import hydra
from omegaconf import OmegaConf

from . import distrib
from .wav import get_wav_datasets, get_musdb_wav_datasets
from .demucs import Demucs
from .hdemucs import HDemucs
from .htdemucs import HTDemucs
from .repitch import RepitchedWrapper
from .solver import Solver
from .states import capture_init
from .utils import random_subset
from .drum_datasets import get_drum_datasets, DrumDataset
from .drum_losses import DrumPatternLoss

logger = logging.getLogger(__name__)

TRAINING_FOLDERS = [
    '/DATA/Training Data/organized drums',
    '/DATA/Training Data/beat stems',
    '/DATA/Training Data/song drum stems'
]

class TorchHDemucsWrapper(nn.Module):
    """Wrapper around torchaudio HDemucs implementation to provide the proper metadata for model evaluation."""
    
    @capture_init
    def __init__(self, **kwargs):
        super().__init__()
        from torchaudio.models import HDemucs as TorchHDemucs
        self.samplerate = kwargs.pop('samplerate')
        self.segment = kwargs.pop('segment')
        self.sources = kwargs['sources']
        self.torch_hdemucs = TorchHDemucs(**kwargs)

    def forward(self, mix):
        return self.torch_hdemucs(mix)


class DrumPatternWrapper(nn.Module):
    """Memory-efficient wrapper for drum pattern recognition."""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        # Reduced channel dimensions for memory efficiency
        self.pattern_heads = nn.ModuleDict({
            source: nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=7, padding=3, stride=2),  # Reduced channels and kernel
                nn.BatchNorm1d(32),  # Added BatchNorm for better training stability
                nn.ReLU(),
                nn.Conv1d(32, 16, kernel_size=5, padding=2, stride=2),  # Added stride for downsampling
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Conv1d(16, 1, kernel_size=3, padding=1),
                nn.Sigmoid()  # Added sigmoid for pattern probability
            ) for source in base_model.sources
        })
        
    @torch.cuda.amp.autocast()  # Use automatic mixed precision
    def forward(self, mix):
        """Forward pass with memory-efficient processing."""
        # Get the primary source separation output
        sources = self.base_model(mix)
        
        # Process patterns in chunks if the audio is long
        if sources.shape[-1] > 44100 * 10:  # If longer than 10 seconds
            patterns = self._chunked_pattern_recognition(sources)
        else:
            patterns = self._single_pattern_recognition(sources)
            
        return sources, patterns
    
    @torch.cuda.amp.autocast()
    def _single_pattern_recognition(self, sources):
        """Process patterns for shorter audio segments."""
        patterns = {}
        for idx, source in enumerate(self.base_model.sources):
            source_audio = sources[:, idx:idx+1]
            # Normalize input to pattern recognition
            source_audio = source_audio / (torch.max(torch.abs(source_audio)) + 1e-8)
            patterns[source] = self.pattern_heads[source](source_audio)
        return patterns
    
    @torch.cuda.amp.autocast()
    def _chunked_pattern_recognition(self, sources, chunk_size=44100*5):
        """Process patterns in chunks for longer audio."""
        patterns = {source: [] for source in self.base_model.sources}
        chunks = sources.shape[-1] // chunk_size + (1 if sources.shape[-1] % chunk_size else 0)
        
        for i in range(chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, sources.shape[-1])
            
            # Process each source in the current chunk
            for idx, source in enumerate(self.base_model.sources):
                source_audio = sources[:, idx:idx+1, start_idx:end_idx]
                # Normalize input to pattern recognition
                source_audio = source_audio / (torch.max(torch.abs(source_audio)) + 1e-8)
                chunk_pattern = self.pattern_heads[source](source_audio)
                patterns[source].append(chunk_pattern)
        
        # Concatenate chunks for each source
        return {source: torch.cat(pattern_chunks, dim=-1) 
               for source, pattern_chunks in patterns.items()}


def get_model(args):
    """Initializes the model based on specified configuration."""
    extra = {
        'sources': list(args.dset.sources),
        'audio_channels': args.dset.channels,
        'samplerate': args.dset.samplerate,
        'segment': args.model_segment or 4 * args.dset.segment,
        'use_pattern_recognition': args.model.get('use_pattern_recognition', True),
        'drum_specific_layers': args.model.get('drum_specific_layers', True)
    }
    klass = {
        'demucs': Demucs,
        'hdemucs': HDemucs,
        'htdemucs': HTDemucs,
        'torch_hdemucs': TorchHDemucsWrapper,
    }.get(args.model)
    if klass is None:
        raise ValueError(f"Model type '{args.model}' is not recognized.")
    
    kw = OmegaConf.to_container(getattr(args, args.model), resolve=True)
    model = klass(**extra, **kw)
    
    if args.model == 'hdemucs' and extra['use_pattern_recognition']:
        model = DrumPatternWrapper(model)
    
    return model


def get_optimizer(model, args):
    """Sets up optimizer based on configuration."""
    seen_params, other_params, groups = set(), [], []
    for n, module in model.named_modules():
        if hasattr(module, "make_optim_group"):
            group = module.make_optim_group()
            assert set(group["params"]).isdisjoint(seen_params)
            seen_params |= set(group["params"])
            groups.append(group)
    other_params = [param for param in model.parameters() if param not in seen_params]
    groups.insert(0, {"params": other_params})

    optimizers = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }
    optimizer_cls = optimizers.get(args.optim.optim.lower())
    if optimizer_cls is None:
        raise ValueError(f"Invalid optimizer type '{args.optim.optim}'")
    return optimizer_cls(groups, lr=args.optim.lr, betas=(args.optim.momentum, args.optim.beta2), weight_decay=args.optim.weight_decay)


def get_datasets(args):
    """Creates train and validation datasets."""
    if args.dset.backend:
        torchaudio.set_audio_backend(args.dset.backend)

    # If using drum-specific dataset
    if args.model == 'hdemucs' and hasattr(args.dset, 'use_drums') and args.dset.use_drums:
        return get_drum_datasets(args)

    # Original dataset loading logic for other cases
    train_set, valid_set = (get_musdb_wav_datasets(args.dset) if args.dset.use_musdb else ([], []))
    if args.dset.wav:
        extra_train_set, extra_valid_set = get_wav_datasets(args.dset)
        train_set, valid_set = torch.utils.data.ConcatDataset([train_set, extra_train_set]), torch.utils.data.ConcatDataset([valid_set, extra_valid_set])

    if args.dset.wav2:
        extra_train_set, extra_valid_set = get_wav_datasets(args.dset, "wav2")
        reps = max(1, round(len(extra_train_set) / len(train_set) * (1 / args.dset.wav2_weight - 1))) if args.dset.wav2_weight else 1
        train_set = torch.utils.data.ConcatDataset([train_set] * reps + [extra_train_set])
        if args.dset.wav2_valid and args.dset.wav2_weight:
            valid_set = torch.utils.data.ConcatDataset([valid_set, random_subset(extra_valid_set, int(round(args.dset.wav2_weight * len(valid_set) / (1 - args.dset.wav2_weight))))])

    if args.dset.valid_samples:
        valid_set = random_subset(valid_set, args.dset.valid_samples)
    return train_set, valid_set


def get_drum_datasets(args):
    """Creates train and validation datasets for drum separation."""
    sample_dirs = [
        '/DATA/Training Data/organized drums'
    ]
    
    stem_dirs = [
        '/DATA/Training Data/song drum stems',
        '/DATA/Training Data/beat stems'
    ]
    
    train_dataset = DrumDataset(
        sample_dirs=sample_dirs,
        stem_dirs=stem_dirs,
        sources=args.dset.sources,
        sample_rate=args.dset.samplerate,
        segment_duration=args.dset.segment,
        channels=args.dset.channels,
        normalize=args.dset.normalize,
        sample_ratio=args.dset.sample_ratio
    )
    
    # Create a smaller validation set (10% of training data)
    valid_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - valid_size
    train_set, valid_set = torch.utils.data.random_split(
        train_dataset, 
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    return train_set, valid_set


def get_solver(args, model_only=False):
    """Creates the solver for training the model."""
    if model_only:
        return get_model(args)

    # Initialize model and optimizer with proper device placement
    model = get_model(args)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = get_optimizer(model, args)

    # Get training and validation datasets
    train_set, valid_set = get_datasets(args)
    loaders = {}
    
    if train_set is not None:
        loaders['train'] = distrib.loader(
            train_set, batch_size=args.batch_size, shuffle=True,
            num_workers=args.misc.num_workers, drop_last=True)
    if valid_set is not None:
        loaders['valid'] = distrib.loader(
            valid_set, batch_size=args.batch_size, shuffle=False,
            num_workers=args.misc.num_workers, drop_last=True)

    # Initialize the solver
    solver = Solver(loaders, model, optimizer, args)
    
    # Add drum-specific loss components if using pattern recognition
    if hasattr(model, 'pattern_heads'):
        solver.pattern_loss = DrumPatternLoss(
            alpha=args.loss.get('pattern_weight', 0.3),
            use_onset_loss=args.loss.get('use_onset_loss', True),
            use_pattern_loss=args.loss.get('use_pattern_loss', True)
        )
        if torch.cuda.is_available():
            solver.pattern_loss = solver.pattern_loss.cuda()
    
    # Set up mixed precision training if available
    if torch.cuda.is_available():
        solver.scaler = torch.cuda.amp.GradScaler()
    
    return solver


def load_custom_data(training_folders, batch_size):
    """
    Function to set up the dataloader for training data across the specified folders
    with variable sample rates.
    """
    print("Loading data...")
    datasets = []
    for folder in training_folders:
        if os.path.exists(folder):
            print(f"Loading data from: {folder}")
            # Initialize the dataset for each folder
            dataset = Dataset(
                root=folder,
                streams=["drums"],  # Focus on drums
                channels=2,         # Stereo audio
                duration=10,        # Segment duration in seconds
                samplerate=None,    # Use the sample rate of the files (no resampling)
                stride=5,           # Overlap between segments
                augment=True        # Apply data augmentation
            )
            datasets.append(dataset)
        else:
            print(f"Folder not found: {folder}")
    
    if datasets:
        print("Combining datasets...")
        combined_dataset = torch.utils.data.ConcatDataset(datasets)
        dataloader = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        return dataloader
    else:
        raise RuntimeError("No datasets found in the specified folders.")


@hydra_main(config_path="../conf", config_name="config", version_base="1.1")
def main(args):
    for attr in ["musdb", "wav", "metadata"]:
        val = getattr(args.dset, attr)
        if val:
            setattr(args.dset, attr, hydra.utils.to_absolute_path(val))

    os.environ["OMP_NUM_THREADS"], os.environ["MKL_NUM_THREADS"] = "1", "1"
    if args.misc.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info(f"For logs, checkpoints, and samples, check {os.getcwd()}")
    logger.debug(args)
    logger.debug(get_xp().cfg)

    solver = get_solver(args)
    try:
        solver.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted. Saving current checkpoint.")
        solver._serialize(epoch=solver.history[-1] if solver.history else 0)


if '_DORA_TEST_PATH' in os.environ:
    main.dora.dir = Path(os.environ['_DORA_TEST_PATH'])

if __name__ == "__main__":
    main()