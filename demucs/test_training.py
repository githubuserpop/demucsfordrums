import unittest
import torch
import torchaudio
import os
import tempfile
from pathlib import Path
from .train import get_model, get_solver
from .drum_datasets import DrumDataset, get_drum_datasets

class DotDict(dict):
    """A dictionary that allows dot notation access to its keys."""
    def __getattr__(self, key):
        if key not in self:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
        return self[key]
    
    def __setattr__(self, key, value):
        self[key] = value

class TestTraining(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sample_dir = os.path.join(self.temp_dir, 'samples')
        self.stem_dir = os.path.join(self.temp_dir, 'stems')
        os.makedirs(self.sample_dir)
        os.makedirs(self.stem_dir)

        # Create dummy audio files
        self.sample_rate = 44100
        self.duration = 4  # seconds
        dummy_audio = torch.randn(self.sample_rate * self.duration)
        
        # Save a few dummy audio files
        for i in range(3):
            sample_path = os.path.join(self.sample_dir, f'sample_{i}.wav')
            stem_path = os.path.join(self.stem_dir, f'stem_{i}.wav')
            torchaudio.save(sample_path, dummy_audio.unsqueeze(0), self.sample_rate)
            torchaudio.save(stem_path, dummy_audio.unsqueeze(0), self.sample_rate)

        self.config = DotDict({
            'dset': DotDict({
                'backend': 'soundfile',
                'sample_dirs': [self.sample_dir],
                'stem_dirs': [self.stem_dir],
                'sample_ratio': 0.9,
                'valid_samples': 1
            }),
            'model': DotDict({
                'channels': 48,
                'growth': 2,
                'depth': 6,
                'kernel_size': 8,
                'stride': 4,
                'normalize': True,
                'resample': True,
                'rescale': 0.1,
                'segment': 4
            }),
            'optim': DotDict({
                'lr': 3e-4,
                'epochs': 100,
                'batch_size': 4
            })
        })

    def test_dataset_creation(self):
        """Test that datasets can be created successfully."""
        train_set, valid_set = get_drum_datasets(self.config)
        self.assertIsNotNone(train_set)
        self.assertIsNotNone(valid_set)
        self.assertGreater(len(train_set), 0)
        self.assertGreater(len(valid_set), 0)

    def test_model_creation(self):
        """Test that model can be created successfully."""
        model = get_model(self.config)
        self.assertIsNotNone(model)
        
        # Test model properties
        base_model = model.base_model if hasattr(model, 'base_model') else model
        self.assertEqual(base_model.channels, self.config.model.channels)
        self.assertEqual(base_model.depth, self.config.model.depth)
        self.assertEqual(base_model.sources, ['drums'])

    def test_solver_creation(self):
        """Test that solver can be created successfully."""
        solver = get_solver(self.config)
        self.assertIsNotNone(solver)
        self.assertIsNotNone(solver.model)
        self.assertIsNotNone(solver.optimizer)

    def test_forward_pass(self):
        """Test that a forward pass through the model works."""
        model = get_model(self.config)
        # Create input tensor with correct shape [batch, channels, time]
        x = torch.randn(4, 2, 44100 * 4)  # 4 seconds of audio at 44.1kHz
        output = model(x)
        self.assertEqual(output.shape[0], 4)  # batch size
        self.assertEqual(output.shape[1], 2)  # number of sources
        self.assertEqual(output.shape[2], 44100 * 4)  # time dimension

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

if __name__ == '__main__':
    unittest.main()
