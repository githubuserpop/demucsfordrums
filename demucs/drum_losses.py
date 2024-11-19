import torch
import torch.nn as nn
import torch.nn.functional as F

class DrumPatternLoss(nn.Module):
    def __init__(self, sample_rate=44100, hop_length=1024):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        
    def forward(self, pred, target, drum_type):
        pred_spec = torch.stft(pred, n_fft=2048, hop_length=self.hop_length, 
                             return_complex=True)
        target_spec = torch.stft(target, n_fft=2048, hop_length=self.hop_length, 
                               return_complex=True)
        
        # Basic reconstruction loss
        recon_loss = F.mse_loss(pred_spec.abs(), target_spec.abs())
        
        # Pattern-specific losses
        if drum_type == 'kick':
            pattern_loss = self._beat_consistency_loss(pred_spec, target_spec, 
                                                     freq_range=(20, 200))
            timing_loss = self._onset_timing_loss(pred_spec, target_spec)
            return recon_loss + 0.5 * pattern_loss + 0.3 * timing_loss
            
        elif drum_type in ['snare', 'clap']:
            pattern_loss = self._transient_consistency_loss(pred_spec, target_spec, 
                                                          freq_range=(200, 2000))
            return recon_loss + 0.4 * pattern_loss
            
        else:  # hi-hats and percussion
            pattern_loss = self._rhythm_regularity_loss(pred_spec, target_spec, 
                                                      freq_range=(2000, 8000))
            return recon_loss + 0.3 * pattern_loss