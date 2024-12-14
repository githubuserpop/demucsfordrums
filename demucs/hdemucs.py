# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
This code contains the optimized spectrogram and Hybrid version of Demucs.
Includes memory-efficient processing and reduced complexity architecture.
"""
from copy import deepcopy
import math
import typing as tp

from openunmix.filtering import wiener
import torch
from torch import nn
from torch.nn import functional as F

from .demucs import DConv, rescale_module
from .states import capture_init
from .spec import spectro, ispectro

def pad1d(x: torch.Tensor, paddings: tp.Tuple[int, int], mode: str = 'constant', value: float = 0.):
    """Memory-efficient padding implementation."""
    x0 = x
    length = x.shape[-1]
    padding_left, padding_right = paddings
    if mode == 'reflect':
        max_pad = max(padding_left, padding_right)
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            extra_pad_right = min(padding_right, extra_pad)
            extra_pad_left = extra_pad - extra_pad_right
            paddings = (padding_left - extra_pad_left, padding_right - extra_pad_right)
            x = F.pad(x, (extra_pad_left, extra_pad_right))
    out = F.pad(x, paddings, mode, value)
    assert out.shape[-1] == length + padding_left + padding_right
    return out

class EfficientHEncLayer(nn.Module):
    """Memory-efficient encoder layer using depthwise separable convolutions."""
    def __init__(self, chin, chout, kernel_size=8, stride=4, norm_groups=1,
                 freq=True, context=0, pad=True):
        super().__init__()
        self.freq = freq
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        padding = kernel_size // 4 if pad else 0

        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            padding = [padding, 0]
            # Depthwise separable convolution for frequency domain
            self.conv = nn.Sequential(
                nn.Conv2d(chin, chin, kernel_size, stride=stride, padding=padding, groups=chin),
                nn.Conv2d(chin, chout, 1)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(chin, chin, kernel_size, stride=stride, padding=padding, groups=chin),
                nn.Conv1d(chin, chout, 1)
            )

        self.norm = nn.GroupNorm(norm_groups, chout) if norm_groups > 0 else nn.Identity()
        self.context = context
        if context:
            self.ctx_conv = nn.Conv2d(chout, chout, [1, context], padding=[0, context//2])

    def forward(self, x, inject=None):
        y = self.conv(x)
        y = self.norm(y)
        if self.context:
            y = self.ctx_conv(y)
        if inject is not None:
            y = y + inject
        return y

class EfficientHDecLayer(nn.Module):
    """Memory-efficient decoder layer using depthwise separable convolutions."""
    def __init__(self, chin, chout, kernel_size=8, stride=4, norm_groups=1,
                 freq=True, context=1, pad=True):
        super().__init__()
        self.freq = freq
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad_size = kernel_size // 4 if pad else 0

        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            padding = [self.pad_size, 0]
            # Depthwise separable transposed convolution
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(chin, chin, kernel_size, stride=stride, padding=padding, groups=chin),
                nn.Conv2d(chin, chout, 1)
            )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose1d(chin, chin, kernel_size, stride=stride, padding=self.pad_size, groups=chin),
                nn.Conv1d(chin, chout, 1)
            )

        self.norm = nn.GroupNorm(norm_groups, chout) if norm_groups > 0 else nn.Identity()

    def forward(self, x):
        y = self.conv(x)
        y = self.norm(y)
        return y

class EfficientMultiWrap(nn.Module):
    """Efficient wrapper for multi-band processing."""
    def __init__(self, layer, **kwargs):
        super().__init__()
        if not isinstance(layer, nn.Module):
            raise TypeError("Layer must be a PyTorch Module")
        
        # Validate that the layer has expected attributes
        if not hasattr(layer, 'forward'):
            raise AttributeError("Layer must have a forward method")
            
        # Store original layer dimensions if available
        self.in_channels = getattr(layer, 'in_channels', None)
        self.out_channels = getattr(layer, 'out_channels', None)
        
        self.layer = layer

    def forward(self, x_band):
        # Input validation
        if not isinstance(x_band, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
            
        if x_band.dim() not in [3, 4]:  # Allow both 3D and 4D tensors
            raise ValueError(f"Expected 3D or 4D tensor, got {x_band.dim()}D")
            
        # Check input channels if known
        if self.in_channels is not None:
            if x_band.size(1) != self.in_channels:
                raise ValueError(f"Expected {self.in_channels} input channels, got {x_band.size(1)}")
        
        try:
            output = self.layer(x_band)
            
            # Validate output
            if not isinstance(output, torch.Tensor):
                raise TypeError("Layer output must be a torch.Tensor")
                
            # Check output channels if known
            if self.out_channels is not None:
                if output.size(1) != self.out_channels:
                    raise ValueError(f"Expected {self.out_channels} output channels, got {output.size(1)}")
                    
            return output
            
        except Exception as e:
            raise RuntimeError(f"Error in EfficientMultiWrap forward pass: {str(e)}")

class HEncLayer(nn.Module):
    """Hybrid encoder layer."""
    def __init__(self, chin, chout, kernel_size=8, stride=4, norm_groups=1, freq=True):
        super().__init__()
        self.freq = freq
        self.kernel_size = kernel_size
        self.stride = stride
        padding = kernel_size // 4

        if freq:
            self.conv = nn.Conv2d(chin, chout, kernel_size=(kernel_size, 1), 
                                stride=(stride, 1), padding=(padding, 0))
        else:
            self.conv = nn.Conv1d(chin, chout, kernel_size, 
                                stride=stride, padding=padding)
        
        self.norm = nn.GroupNorm(norm_groups, chout) if norm_groups > 0 else nn.Identity()
        self.act = nn.GLU(1) if freq else nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class MultiWrap(nn.Module):
    """Wrapper for handling multiple band processing."""
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        B, C, Fr, T = x.shape
        x = x.view(-1, 1, Fr, T)
        x = self.module(x)
        x = x.view(B, -1, x.shape[-2], x.shape[-1])
        return x

class HDecLayer(nn.Module):
    """Hybrid decoder layer."""
    def __init__(self, chin, chout, kernel_size=8, stride=4, norm_groups=1, 
                 freq=True, context=1):
        super().__init__()
        self.freq = freq
        self.kernel_size = kernel_size
        self.stride = stride
        padding = kernel_size // 4

        if freq:
            self.conv = nn.ConvTranspose2d(chin, chout, kernel_size=(kernel_size, 1),
                                         stride=(stride, 1), padding=(padding, 0))
        else:
            self.conv = nn.ConvTranspose1d(chin, chout, kernel_size,
                                         stride=stride, padding=padding)
        
        self.norm = nn.GroupNorm(norm_groups, chout) if norm_groups > 0 else nn.Identity()
        self.act = nn.GLU(1) if freq else nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class ScaledEmbedding(nn.Module):
    """Efficient scaled embedding layer."""
    def __init__(self, num_embeddings: int, embedding_dim: int, scale: float = 1.):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.scale = scale

    def forward(self, x):
        out = self.embedding(x)
        return out * self.scale

class HDemucs(nn.Module):
    """Optimized Hybrid Demucs architecture for drum separation."""
    def __init__(self, channels=32, growth=1.5, nfft=2048, 
                 depth=4, freq_emb_dim=16, normalize=True,
                 chunk_size=262144, efficient_mode=True):
        """Initialize the optimized HDemucs model."""
        super(HDemucs, self).__init__()
        self.channels = channels
        self.depth = depth
        self.nfft = nfft
        self.normalize = normalize
        self.chunk_size = chunk_size
        self.efficient_mode = efficient_mode
        self.floor = 1e-8
        self.sources = ['drums']  # For drum separation

        # Lightweight frequency embedding
        self.freq_emb = nn.Sequential(
            nn.Linear(1, freq_emb_dim),
            nn.ReLU(),
            nn.Linear(freq_emb_dim, channels)
        )

        self.channel_proj = nn.Conv2d(2, channels, 1)

        # Initialize decoders with efficient layers
        self.kick_decoder = nn.ModuleList()
        self.snare_decoder = nn.ModuleList()
        self.hihat_decoder = nn.ModuleList()
        self.clap_decoder = nn.ModuleList()
        self.openhat_decoder = nn.ModuleList()
        self.perc_decoder = nn.ModuleList()

        chin = channels
        for index in range(depth):
            chout = int(channels * (growth ** index))
            freq = True

            def create_decoder_layer(chin, chout, freq):
                try:
                    dec_layer = EfficientHDecLayer(chin, chout, freq=freq)
                    # Store channel information for validation
                    dec_layer.in_channels = chin
                    dec_layer.out_channels = chout
                    return EfficientMultiWrap(dec_layer)
                except Exception as e:
                    raise RuntimeError(f"Failed to create decoder layer: {str(e)}")

            # Create decoder layers with proper error handling
            decoder_layer = lambda: create_decoder_layer(chin, chout, freq)

            self.kick_decoder.append(decoder_layer())
            self.snare_decoder.append(decoder_layer())
            self.hihat_decoder.append(decoder_layer())
            self.clap_decoder.append(decoder_layer())
            self.openhat_decoder.append(decoder_layer())
            self.perc_decoder.append(decoder_layer())

            chin = chout

    def forward(self, mix):
        """Forward pass with optional chunk processing."""
        if self.efficient_mode and mix.shape[-1] > self.chunk_size:
            return self._forward_chunks(mix)
        return self._forward_chunk(mix)

    def _forward_chunk(self, x):
        """Process a single chunk."""
        # Apply STFT
        spec = self._spec(x)
        B, C, Fr, T = spec.shape

        # Apply frequency embedding
        freq_emb = torch.linspace(0, 1, Fr, device=x.device).view(1, 1, -1, 1)
        freq_emb = self.freq_emb(freq_emb)
        spec = spec + freq_emb

        # Project to model channels
        x = self.channel_proj(spec)

        # Process through decoders
        outputs = {}
        decoders = {
            'kicks': self.kick_decoder,
            'snares': self.snare_decoder,
            'hi_hats': self.hihat_decoder,
            'claps': self.clap_decoder,
            'open_hats': self.openhat_decoder,
            'percs': self.perc_decoder
        }

        for name, decoder in decoders.items():
            out = x
            for layer in decoder:
                out = layer(out)
            outputs[name] = out

        return outputs

    def _forward_chunks(self, mix):
        """Process audio in chunks for memory efficiency."""
        chunk_size = self.chunk_size
        chunks = mix.unfold(-1, chunk_size, chunk_size // 2)
        results = []

        for chunk in chunks.transpose(0, -1):
            with torch.no_grad():
                result = self._forward_chunk(chunk)
            results.append(result)

        # Combine chunks
        output = {}
        for key in results[0].keys():
            chunks = torch.stack([r[key] for r in results], dim=-1)
            output[key] = chunks.mean(dim=-1)

        return output

    def _spec(self, x):
        """Efficient STFT implementation."""
        spec = torch.stft(
            x, 
            n_fft=self.nfft,
            hop_length=self.nfft // 4,
            window=torch.hann_window(self.nfft, device=x.device),
            return_complex=True
        )
        spec = torch.abs(spec)
        if self.normalize:
            spec = spec / (spec.mean() + self.floor)
        return spec