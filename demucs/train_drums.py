import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from pathlib import Path
from .hdemucs import HDemucs
from .solver import Solver
from .drum_datasets import DrumDataset, get_drum_datasets
import hydra
import logging
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from functools import partial

logger = logging.getLogger(__name__)

@dataclass
class DatasetConfig:
    musdb: Optional[str] = None
    samplerate: int = 44100
    channels: int = 2
    sources: List[str] = field(default_factory=lambda: ["kick", "clap", "snare", "hihat", "openhat", "perc"])
    segment: int = 10
    shift: int = 1
    normalize: bool = True
    train_valid: bool = True
    download: bool = False
    sample_dirs: List[str] = field(default_factory=lambda: ['/DATA/Training Data/organized drums'])
    stem_dirs: List[str] = field(default_factory=lambda: [
        '/DATA/Training Data/song drum stems',
        '/DATA/Training Data/beat stems'
    ])
    sample_ratio: float = 0.3
    batch_size: int = 4

@dataclass
class OptimConfig:
    name: str = 'adam'
    loss: str = 'l1'
    optimizer: str = 'adam'
    momentum: float = 0.9
    clip_grad: float = 0.5

@dataclass
class AugmentConfig:
    shift_same: bool = True
    flip: bool = True
    scale: Dict = field(default_factory=lambda: {'proba': 0.5, 'min': 0.3, 'max': 1.2})
    remix: Dict = field(default_factory=lambda: {'proba': 0.5, 'min': 0.4, 'max': 1.0})

@dataclass
class MiscConfig:
    num_workers: int = 4
    verbose: bool = True
    show_progress: bool = True

@dataclass
class EMAConfig:
    batch: List[float] = field(default_factory=lambda: [0.9999, 0.9999])
    epoch: List[float] = field(default_factory=list)

@dataclass
class TestConfig:
    every: int = 5
    workers: int = 2
    metric: str = 'loss'
    sdr: bool = True
    shifts: int = 1
    overlap: float = 0.25
    best: bool = True

@dataclass
class DrumTrainingConfig:
    dset: DatasetConfig = field(default_factory=DatasetConfig)
    epochs: int = 100
    batch_size: int = 4
    lr: float = 3e-4
    optim: OptimConfig = field(default_factory=OptimConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    misc: MiscConfig = field(default_factory=MiscConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    test: TestConfig = field(default_factory=TestConfig)
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')
    seed: int = 42
    quant: Optional[str] = None

def setup_drum_model(args):
    """Initialize a drum-specific HDemucs model."""
    model = HDemucs(
        channels=48,  # Base number of channels
        growth=1.5,  # Channel growth factor
        nfft=2048,  # FFT size
        depth=4,  # Number of layers
        freq_emb_dim=16,  # Frequency embedding dimension
        normalize=True,
        chunk_size=262144,
        efficient_mode=True
    )
    return model

# Register Hydra configuration
cs = ConfigStore.instance()
cs.store(name="config", node=DrumTrainingConfig)

@hydra.main(version_base=None, config_name="config")
def train_drum_model(cfg: DrumTrainingConfig):
    """Train a drum separation model."""
    logger.info("Initializing drum separation training...")
    
    try:
        # Set random seed for reproducibility
        torch.manual_seed(cfg.seed)
        
        # Initialize model and optimizer
        model = setup_drum_model(cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        
        # Add batch_size to dataset config
        cfg.dset.batch_size = cfg.batch_size
        
        # Initialize solver with datasets
        solver = Solver(None, model, optimizer, cfg)
        
        # Train model
        solver.train()
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    train_drum_model()
