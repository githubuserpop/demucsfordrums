# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Data augmentations.
"""

import random
import torch as th
from torch import nn


class Shift(nn.Module):
    """
    Randomly shift audio in time by up to `shift` samples.
    """
    def __init__(self, shift=8192, same=False):
        super().__init__()
        self.shift = shift
        self.same = same

    def forward(self, wav):
        batch, sources, channels, time = wav.size()
        length = time - self.shift
        if self.shift > 0:
            if not self.training:
                wav = wav[..., :length]
            else:
                srcs = 1 if self.same else sources
                offsets = th.randint(self.shift, [batch, srcs, 1, 1], device=wav.device)
                offsets = offsets.expand(-1, sources, channels, -1)
                indexes = th.arange(length, device=wav.device)
                wav = wav.gather(3, indexes + offsets)
        return wav


class FlipChannels(nn.Module):
    """
    Flip left-right channels.
    """
    def forward(self, wav):
        batch, sources, channels, time = wav.size()
        if self.training and wav.size(2) == 2:
            left = th.randint(2, (batch, sources, 1, 1), device=wav.device)
            left = left.expand(-1, -1, -1, time)
            right = 1 - left
            wav = th.cat([wav.gather(2, left), wav.gather(2, right)], dim=2)
        return wav


class FlipSign(nn.Module):
    """
    Random sign flip.
    """
    def forward(self, wav):
        batch, sources, channels, time = wav.size()
        if self.training:
            signs = th.randint(2, (batch, sources, 1, 1), device=wav.device, dtype=th.float32)
            wav = wav * (2 * signs - 1)
        return wav


class Remix(nn.Module):
    """
    Shuffle sources to make new mixes.
    """
    def __init__(self, proba=1, group_size=4):
        """
        Shuffle sources within one batch.
        Each batch is divided into groups of size `group_size` and shuffling is done within
        each group separatly. This allow to keep the same probability distribution no matter
        the number of GPUs. Without this grouping, using more GPUs would lead to a higher
        probability of keeping two sources from the same track together which can impact
        performance.
        """
        super().__init__()
        self.proba = proba
        self.group_size = group_size

    def forward(self, wav):
        batch, streams, channels, time = wav.size()
        device = wav.device

        if self.training and random.random() < self.proba:
            group_size = self.group_size or batch
            if batch % group_size != 0:
                raise ValueError(f"Batch size {batch} must be divisible by group size {group_size}")
            groups = batch // group_size
            wav = wav.view(groups, group_size, streams, channels, time)
            permutations = th.argsort(th.rand(groups, group_size, streams, 1, 1, device=device),
                                      dim=1)
            wav = wav.gather(1, permutations.expand(-1, -1, -1, channels, time))
            wav = wav.view(batch, streams, channels, time)
        return wav


class Scale(nn.Module):
    def __init__(self, proba=1., min=0.25, max=1.25):
        super().__init__()
        self.proba = proba
        self.min = min
        self.max = max

    def forward(self, wav):
        batch, streams, channels, time = wav.size()
        device = wav.device
        if self.training and random.random() < self.proba:
            scales = th.empty(batch, streams, 1, 1, device=device).uniform_(self.min, self.max)
            wav *= scales
        return wav

class DrumPatternShift(nn.Module):
    """
    Intelligent drum pattern shift that preserves rhythmic structure
    """
    def __init__(self, shift=8192, tempo_aware=True, pattern_length=4):
        super().__init__()
        self.shift = shift
        self.tempo_aware = tempo_aware
        self.pattern_length = pattern_length  # in beats
        
    def forward(self, wav):
        batch, sources, channels, time = wav.size()
        if not self.training:
            return wav[..., :time-self.shift]
            
        # Calculate beat-aligned shifts for each drum type
        shifts = []
        for source in range(sources):
            if source == 0:  # Kick
                # Align shifts to measure boundaries
                shift = th.randint(0, self.pattern_length, (batch, 1, 1, 1))
                shift = shift * (time // self.pattern_length)
            elif source in [1, 2]:  # Snare/Clap
                # Allow for backbeat variations
                shift = th.randint(0, self.pattern_length // 2, (batch, 1, 1, 1))
                shift = shift * (time // (self.pattern_length // 2))
            else:  # Hi-hats/Percussion
                # Finer-grained shifts for more variation
                shift = th.randint(0, self.shift, (batch, 1, 1, 1))
            shifts.append(shift)
            
        shifts = th.cat(shifts, dim=1).to(wav.device)
        indexes = th.arange(time - self.shift, device=wav.device)
        return wav.gather(3, indexes + shifts)


class DrumFrequencyAugment(nn.Module):
    """
    Frequency-based augmentation specific to drum characteristics
    """
    def __init__(self, intensity=0.1):
        super().__init__()
        self.intensity = intensity
        
    def forward(self, wav):
        if not self.training:
            return wav
            
        batch, sources, channels, time = wav.size()
        
        # Apply frequency-specific augmentations per drum type
        for source in range(sources):
            if source == 0:  # Kick
                # Enhance low frequencies (50-100 Hz)
                wav[:, source] = self._boost_frequencies(
                    wav[:, source], low_cut=50, high_cut=100)
            elif source in [1, 2]:  # Snare/Clap
                # Enhance attack transients (200-1000 Hz)
                wav[:, source] = self._boost_transients(
                    wav[:, source], freq_range=(200, 1000))
            else:  # Hi-hats/Percussion
                # Enhance high frequencies (5000-10000 Hz)
                wav[:, source] = self._boost_frequencies(
                    wav[:, source], low_cut=5000, high_cut=10000)
                
        return wav
    
    def _boost_frequencies(self, audio, low_cut, high_cut):
        # FFT-based frequency boost
        fft = th.fft.rfft(audio, dim=-1)
        freqs = th.fft.rfftfreq(audio.shape[-1])
        mask = ((freqs >= low_cut) & (freqs <= high_cut)).float()
        fft *= (1 + self.intensity * mask)
        return th.fft.irfft(fft, dim=-1)
    
    def _boost_transients(self, audio, freq_range):
        # Transient detection and enhancement
        envelope = th.abs(audio)
        peaks = (envelope > th.roll(envelope, 1, -1)) & (envelope > th.roll(envelope, -1, -1))
        audio[peaks] *= (1 + self.intensity)
        return audio


class DrumPatternMix(Remix):
    """
    Enhanced version of Remix specifically for drum patterns
    """
    def __init__(self, proba=0.5, group_size=4, pattern_aware=True):
        super().__init__(proba, group_size)
        self.pattern_aware = pattern_aware
        
    def forward(self, wav):
        if not self.training or random.random() >= self.proba:
            return wav
            
        batch, streams, channels, time = wav.size()
        device = wav.device
        
        # Group similar drum types together for more realistic mixing
        groups = {
            'kicks': [0],
            'snares': [1, 2],
            'cymbals': [3, 4],
            'percussion': [5]
        }
        
        # Perform group-wise mixing
        output = wav.clone()
        for group_idx, source_indices in groups.items():
            group_wav = wav[:, source_indices]
            if len(source_indices) > 1:
                # Mix within groups
                permutations = th.randperm(len(source_indices))
                output[:, source_indices] = group_wav[:, permutations]
                
        return output