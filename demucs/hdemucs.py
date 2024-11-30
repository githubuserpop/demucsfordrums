# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
This code contains the spectrogram and Hybrid version of Demucs.
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
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right before the reflection happen."""
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
    assert (out[..., padding_left: padding_left + length] == x0).all()
    return out


class ScaledEmbedding(nn.Module):
    """
    Boost learning rate for embeddings (with `scale`).
    Also, can make embeddings continuous with `smooth`.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 scale: float = 10., smooth=False):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        if smooth:
            weight = torch.cumsum(self.embedding.weight.data, dim=0)
            # when summing gaussian, overscale raises as sqrt(n), so we nornalize by that.
            weight = weight / torch.arange(1, num_embeddings + 1).to(weight).sqrt()[:, None]
            self.embedding.weight.data[:] = weight
        self.embedding.weight.data /= scale
        self.scale = scale

    @property
    def weight(self):
        return self.embedding.weight * self.scale

    def forward(self, x):
        out = self.embedding(x) * self.scale
        return out


class HEncLayer(nn.Module):
    def __init__(self, chin, chout, kernel_size=8, stride=4, norm_groups=1, empty=False,
                 freq=True, dconv=True, norm=True, context=0, dconv_kw={}, pad=True,
                 rewrite=True):
        """Encoder layer. This used both by the time and the frequency branch.

        Args:
            chin: number of input channels.
            chout: number of output channels.
            norm_groups: number of groups for group norm.
            empty: used to make a layer with just the first conv. this is used
                before merging the time and freq. branches.
            freq: this is acting on frequencies.
            dconv: insert DConv residual branches.
            norm: use GroupNorm.
            context: context size for the 1x1 conv.
            dconv_kw: list of kwargs for the DConv class.
            pad: pad the input. Padding is done so that the output size is
                always the input size / stride.
            rewrite: add 1x1 conv at the end of the layer.
        """
        super().__init__()
        norm_fn = lambda d: nn.Identity()  # noqa
        if norm:
            norm_fn = lambda d: nn.GroupNorm(norm_groups, d)  # noqa
        if pad:
            pad = kernel_size // 4
        else:
            pad = 0
        self.pad = pad
        self.freq = freq
        self.kernel_size = kernel_size
        self.stride = stride
        self.empty = empty
        self.norm = norm
        self.context = context
        self.pad = pad
        self.last = False

        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            pad = [pad, 0]
            self.conv = nn.Conv2d(chin, chout, kernel_size, stride=stride, padding=pad)
        else:
            self.conv = nn.Conv1d(chin, chout, kernel_size, stride=stride, padding=pad)

        if norm:
            self.norm1 = norm_fn(chout)
        else:
            self.norm1 = nn.Identity()

        self.rewrite = None
        if rewrite:
            self.rewrite = nn.Conv2d(chout, 2 * chout, 1 + 2 * context, 1, context)
            if norm:
                self.norm2 = norm_fn(2 * chout)

        self.dconv = None
        if dconv:
            self.dconv = DConv(chout, **dconv_kw)

    def forward(self, x, inject=None):
        """
        `inject` is used to inject the result from the time branch into the frequency branch,
        when both have the same stride.
        """
        if not self.freq and x.dim() == 4:
            B, C, Fr, T = x.shape
            x = x.view(B, -1, T)

        if not self.freq:
            le = x.shape[-1]
            if not le % self.stride == 0:
                x = F.pad(x, (0, self.stride - (le % self.stride)))
        y = self.conv(x)
        if self.empty:
            return y
        if inject is not None:
            assert inject.shape[-1] == y.shape[-1], (inject.shape, y.shape)
            if inject.dim() == 3 and y.dim() == 4:
                inject = inject[:, :, None]
            y = y + inject
        y = F.gelu(self.norm1(y))
        if self.dconv:
            if self.freq:
                B, C, Fr, T = y.shape
                y = y.permute(0, 2, 1, 3).reshape(-1, C, T)
            y = self.dconv(y)
            if self.freq:
                y = y.view(B, Fr, C, T).permute(0, 2, 1, 3)
        if self.rewrite:
            z = self.norm2(self.rewrite(y))
            z = F.glu(z, dim=1)
        else:
            z = y
        return z


class HDecLayer(nn.Module):
    def __init__(self, chin, chout, kernel_size=8, stride=4, norm_groups=1, empty=False,
                 freq=True, norm=True, context=1, dconv=True, pad=True, last=False,
                 context_freq=False, **kwargs):
        super().__init__()
        self.chin = chin
        self.chout = chout
        self.freq = freq
        self.kernel_size = kernel_size
        self.stride = stride
        self.empty = empty
        self.norm = norm
        self.context = context
        self.context_freq = context_freq
        self.last = last

        # Calculate padding size
        if isinstance(pad, bool):
            self.pad_size = kernel_size // 4 if pad else 0
        else:
            self.pad_size = pad

        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            padding = [self.pad_size, 0]
            self.conv = nn.ConvTranspose2d(chin, chout, kernel_size, stride=stride, padding=padding)
        else:
            self.conv = nn.ConvTranspose1d(chin, chout, kernel_size, stride=stride, padding=self.pad_size)

        if norm:
            self.norm = nn.GroupNorm(norm_groups, chout)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        """Forward pass for the decoder layer.
        Args:
            x (Tensor): Input tensor [B, C, F, T]
        Returns:
            Tensor: Output tensor [B, C, F', T]
        """
        # Apply transposed convolution
        x = self.conv(x)
        
        # Apply normalization
        x = self.norm(x)
        
        # Apply activation if not the last layer
        if not self.last:
            x = F.gelu(x)
        
        return x


class MultiWrap(nn.Module):
    """Wrap a layer to support multiple frequency bands processing."""
    def __init__(self, layer, **kwargs):
        super().__init__()
        self.layer = layer

    def forward(self, x_band):
        """Forward pass for the wrapped layer.
        Args:
            x_band (Tensor): Input tensor [B, C, F, T]
        Returns:
            Tensor: Output tensor [B, C, F', T]
        """
        return self.layer(x_band)


class HDemucs(nn.Module):
    """Hybrid Demucs architecture for drum separation."""
    def __init__(self, channels=48, growth=2, nfft=4096, 
                 depth=6, freq_emb_dim=32, normalize=True):
        """Initialize the HDemucs model.
        Args:
            channels (int): Initial number of channels
            growth (int): Factor by which the channels grow per layer
            nfft (int): Size of FFT
            depth (int): Number of layers
            freq_emb_dim (int): Dimension of frequency embedding
            normalize (bool): Whether to normalize input
        """
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.nfft = nfft
        self.normalize = normalize
        self.floor = 1e-8
        self.rescale = 0.1

        self.freq_emb = nn.Sequential(
            nn.Linear(1, freq_emb_dim),
            nn.ReLU(),
            nn.Linear(freq_emb_dim, freq_emb_dim),
            nn.ReLU(),
            nn.Linear(freq_emb_dim, channels)
        )

        # Channel projection for input
        self.channel_proj = nn.Conv2d(2, channels, 1)

        # Initialize decoders for each drum component
        self.kick_decoder = nn.ModuleList()
        self.snare_decoder = nn.ModuleList()
        self.hihat_decoder = nn.ModuleList()
        self.clap_decoder = nn.ModuleList()
        self.openhat_decoder = nn.ModuleList()
        self.perc_decoder = nn.ModuleList()

        chin = channels
        for index in range(depth):
            chout = channels * growth
            last = index == depth - 1
            
            # Create decoder layers for each component
            freq = True  # Always True since we're working in frequency domain
            
            # Kick decoder (focused on low frequencies)
            self.kick_decoder.append(
                MultiWrap(HDecLayer(chin, chout, freq=freq, last=last)))

            # Snare decoder (mid frequencies)
            self.snare_decoder.append(
                MultiWrap(HDecLayer(chin, chout, freq=freq, last=last)))

            # Hi-hat decoder (high frequencies)
            self.hihat_decoder.append(
                MultiWrap(HDecLayer(chin, chout, freq=freq, last=last)))

            # Clap decoder (mid-high frequencies)
            self.clap_decoder.append(
                MultiWrap(HDecLayer(chin, chout, freq=freq, last=last)))

            # Open hat decoder (high frequencies)
            self.openhat_decoder.append(
                MultiWrap(HDecLayer(chin, chout, freq=freq, last=last)))

            # Percussion decoder (full frequency range)
            self.perc_decoder.append(
                MultiWrap(HDecLayer(chin, chout, freq=freq, last=last)))

            chin = chout

    def _spec(self, x):
        """Convert input audio to spectrogram.
        Args:
            x (Tensor): Input audio [B, C, T]
        Returns:
            Tensor: Complex spectrogram [B, C, F, T]
        """
        nfft = self.nfft
        hl = nfft // 4
        window = torch.hann_window(nfft).to(x.device)
        
        # Handle input shape
        B, C, T = x.shape
        specs = []
        
        for c in range(C):
            # Compute STFT for each channel
            spec = torch.stft(x[:, c], n_fft=nfft, hop_length=hl,
                            window=window, win_length=nfft,
                            normalized=True, center=True,
                            return_complex=True)  # [B, F, T]
            specs.append(spec)
        
        # Stack channels
        x = torch.stack(specs, dim=1)  # [B, C, F, T]
        
        # Convert to magnitude spectrogram
        return x.abs()

    def forward(self, mix):
        """
        Forward pass for the drum separation model.
        Args:
            mix (Tensor): Input mixture of shape [B, C, T] or [B, T]
        Returns:
            Tensor: Separated drums of shape [B, 6, C, T] where 6 is the number of components
                   (kick, snare, hihat, clap, openhat, perc)
        """
        # Handle 2D input (single channel)
        if mix.dim() == 2:
            mix = mix.unsqueeze(1)
        
        # Store original dimensions
        B, C, T = mix.shape
        
        # Convert to spectrogram
        x = self._spec(mix)  # [B, C, F, T]
        
        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            mean = mono.mean(dim=-1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            x = (x - mean.unsqueeze(-2)) / (self.floor + std.unsqueeze(-2))
        else:
            std = 1
            mean = 0

        # Project input channels to model channels
        x = self.channel_proj(x)  # [B, channels, F, T]

        # Prepare frequency embedding
        freqs = torch.linspace(0, 1, x.shape[2], device=x.device)
        freqs = freqs.view(-1, 1)  # [F, 1]
        freq_emb = self.freq_emb(freqs)  # [F, channels]
        
        # Reshape frequency embedding for addition
        freq_emb = freq_emb.transpose(0, 1)  # [channels, F]
        freq_emb = freq_emb.unsqueeze(0).unsqueeze(-1)  # [1, channels, F, 1]
        freq_emb = freq_emb.expand(B, -1, -1, x.shape[-1])  # [B, channels, F, T]
        
        # Add frequency embedding
        x = x + freq_emb

        # Split into frequency bands for each component
        outputs = []
        for decoder_list in [self.kick_decoder, self.snare_decoder, self.hihat_decoder,
                           self.clap_decoder, self.openhat_decoder, self.perc_decoder]:
            x_band = x
            for layer in decoder_list:
                x_band = layer(x_band)
            outputs.append(x_band)

        # Combine all outputs
        x = torch.stack(outputs, dim=1)  # [B, 6, channels, F, T]
        
        if self.normalize:
            x = x * std.unsqueeze(-2)
            x = x + mean.unsqueeze(-2)
        
        x = x * self.rescale
        
        # Convert back to time domain
        B, D, Ch, F, T = x.shape
        x = x.view(B * D * Ch, F, T)  # Reshape for batch processing
        
        # Convert magnitude to complex
        x = x.to(torch.complex64)  # Convert to complex numbers assuming phase is 0
        
        # Inverse STFT
        nfft = self.nfft
        hl = nfft // 4
        x = torch.istft(x, n_fft=nfft, hop_length=hl,
                       window=torch.hann_window(nfft).to(x.device),
                       win_length=nfft, normalized=True, center=True)
        
        # Reshape back to match input channels
        x = x.view(B, D, C, -1)  # [B, 6, C, T]
        
        # Trim to original length
        x = x[..., :T]
        
        return x