import torch
import torch.nn as nn
import torch.nn.functional as F

class DrumPatternLoss(nn.Module):
    """Memory-efficient loss function for drum separation with pattern recognition."""
    def __init__(self, alpha=0.3, use_onset_loss=True, use_pattern_loss=True):
        super().__init__()
        self.alpha = alpha
        self.use_onset_loss = use_onset_loss
        self.use_pattern_loss = use_pattern_loss
        # Pre-compute STFT parameters for efficiency
        self.n_fft = 1024  # Reduced from 2048
        self.hop_length = 256  # Reduced from 512
        self.register_buffer('window', torch.hann_window(self.n_fft))
        self.sources = ["kick", "snare", "hihat", "toms", "cymbals", "percussion"]

    @torch.cuda.amp.autocast()
    def forward(self, pred_sources, target_sources, pred_patterns=None, target_patterns=None):
        """Memory-efficient forward pass using chunked processing."""
        device = pred_sources.device
        if not hasattr(self, 'window') or self.window.device != device:
            self.register_buffer('window', torch.hann_window(self.n_fft).to(device))

        # Base separation loss (computed in chunks if needed)
        if pred_sources.shape[-1] > 44100 * 10:  # If longer than 10 seconds
            separation_loss = self._chunked_mse_loss(pred_sources, target_sources)
        else:
            separation_loss = F.mse_loss(pred_sources, target_sources)
        
        total_loss = separation_loss
        
        if self.use_onset_loss:
            onset_loss = self._compute_onset_loss(pred_sources, target_sources)
            total_loss = total_loss + self.alpha * onset_loss
            
        if self.use_pattern_loss and pred_patterns is not None and target_patterns is not None:
            pattern_loss = self._compute_pattern_loss(pred_patterns, target_patterns)
            total_loss = total_loss + self.alpha * pattern_loss
            
        return total_loss
    
    def _chunked_mse_loss(self, pred, target, chunk_size=44100*5):
        """Compute MSE loss in chunks to save memory."""
        total_loss = 0
        chunks = pred.shape[-1] // chunk_size + (1 if pred.shape[-1] % chunk_size else 0)
        
        for i in range(chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, pred.shape[-1])
            chunk_loss = F.mse_loss(
                pred[..., start_idx:end_idx],
                target[..., start_idx:end_idx]
            )
            total_loss += chunk_loss
            
        return total_loss / chunks
    
    @torch.cuda.amp.autocast()
    def _compute_onset_loss(self, pred, target):
        """Memory-efficient onset loss computation."""
        onset_loss = 0
        
        for i, source in enumerate(self.sources):
            pred_source = pred[:, i:i+1]
            target_source = target[:, i:i+1]
            
            # Process in chunks if the audio is long
            if pred_source.shape[-1] > 44100 * 10:
                onset_loss += self._chunked_onset_loss(pred_source, target_source)
            else:
                onset_loss += self._single_onset_loss(pred_source, target_source)
            
        return onset_loss / len(self.sources)
    
    def _single_onset_loss(self, pred_source, target_source):
        """Compute onset loss for a single chunk."""
        # Compute spectral flux with pre-computed window
        pred_spec = torch.stft(pred_source.squeeze(1), n_fft=self.n_fft, 
                             hop_length=self.hop_length, window=self.window,
                             return_complex=True).abs()
        target_spec = torch.stft(target_source.squeeze(1), n_fft=self.n_fft,
                               hop_length=self.hop_length, window=self.window,
                               return_complex=True).abs()
        
        # Compute onset envelope (more memory efficient than full flux)
        pred_flux = torch.sum(torch.diff(pred_spec, dim=-1).abs(), dim=1)
        target_flux = torch.sum(torch.diff(target_spec, dim=-1).abs(), dim=1)
        
        return F.mse_loss(pred_flux, target_flux)
    
    def _chunked_onset_loss(self, pred_source, target_source, chunk_size=44100*5):
        """Compute onset loss in chunks for long audio."""
        total_loss = 0
        chunks = pred_source.shape[-1] // chunk_size + (1 if pred_source.shape[-1] % chunk_size else 0)
        
        for i in range(chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, pred_source.shape[-1])
            chunk_loss = self._single_onset_loss(
                pred_source[..., start_idx:end_idx],
                target_source[..., start_idx:end_idx]
            )
            total_loss += chunk_loss
            
        return total_loss / chunks
    
    @torch.cuda.amp.autocast()
    def _compute_pattern_loss(self, pred_patterns, target_patterns):
        """Compute pattern loss with gradient checkpointing if needed."""
        if not pred_patterns or not target_patterns:
            return 0.0
            
        pattern_loss = 0
        for source in pred_patterns:
            if source not in target_patterns:
                continue
            # Use binary cross entropy with reduced precision for memory efficiency
            pred = pred_patterns[source].float()  # Ensure float32
            target = target_patterns[source].float()
            pattern_loss += F.binary_cross_entropy_with_logits(pred, target)
        
        num_patterns = sum(1 for s in pred_patterns if s in target_patterns)
        return pattern_loss / max(num_patterns, 1)