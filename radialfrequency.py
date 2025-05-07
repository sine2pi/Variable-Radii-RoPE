import torch
import torch.nn as nn
import torch.nn.functional as F

class Rotary(nn.Module):
    def __init__(self, dims, max_ctx=1500, learned_freq=True, 
                 use_freq_bands=False, speech_enhanced=False,
                 variable_radius=False, learned_radius=True, init_radius=1.0):
        super().__init__()
        self.dims = dims
        self.use_freq_bands = use_freq_bands
        self.variable_radius = variable_radius
        
        # Configure frequency parameters
        if not use_freq_bands:
            # Original implementation
            self.inv_freq = nn.Parameter(
                1.0 / (10000 ** (torch.arange(0, dims, 2) / dims)),
                requires_grad=learned_freq
            )
            self.bias = nn.Parameter(torch.zeros(max_ctx, dims // 2))
            
            # Global radius parameter (if variable)
            if variable_radius:
                self.radius = nn.Parameter(
                    torch.ones(dims // 2) * init_radius,
                    requires_grad=learned_radius
                )
        else:
            # FrequencyBand implementation
            band_size = dims // 6  # Each band gets 1/3 of dims (x2 for complex numbers)
            
            # Low frequencies (0-500Hz range in speech)
            self.low_freq = nn.Parameter(
                1.0 / (10000 ** (torch.arange(0, band_size, 2) / dims)),
                requires_grad=learned_freq
            )
            
            # Mid frequencies (500-2000Hz in speech)
            self.mid_freq = nn.Parameter(
                1.0 / (10000 ** (torch.arange(band_size, 2*band_size, 2) / dims)),
                requires_grad=learned_freq
            )
            
            # High frequencies (>2000Hz in speech)
            self.high_freq_audio = nn.Parameter(
                1.0 / (10000 ** (torch.arange(2*band_size, 3*band_size, 2) / dims)),
                requires_grad=learned_freq
            )
            
            # Text-specific high frequencies
            self.high_freq_text = nn.Parameter(
                1.0 / (10000 ** (torch.arange(2*band_size, 3*band_size, 2) / dims)),
                requires_grad=learned_freq
            )
            
            # Frequency-specific biases
            if speech_enhanced:
                self.low_bias = nn.Parameter(torch.zeros(max_ctx, band_size // 2))
                self.mid_bias = nn.Parameter(torch.zeros(max_ctx, band_size // 2))
                self.high_bias = nn.Parameter(torch.zeros(max_ctx, band_size // 2))
            else:
                self.bias = nn.Parameter(torch.zeros(max_ctx, dims // 2))
            
            # Band-specific radius parameters (if variable)
            if variable_radius:
                self.low_radius = nn.Parameter(
                    torch.ones(band_size // 2) * init_radius,
                    requires_grad=learned_radius
                )
                self.mid_radius = nn.Parameter(
                    torch.ones(band_size // 2) * init_radius,
                    requires_grad=learned_radius
                )
                self.high_radius_audio = nn.Parameter(
                    torch.ones(band_size // 2) * init_radius,
                    requires_grad=learned_radius
                )
                self.high_radius_text = nn.Parameter(
                    torch.ones(band_size // 2) * init_radius,
                    requires_grad=learned_radius
                )
                
        self.speech_enhanced = speech_enhanced and use_freq_bands

    def forward(self, positions, domain="audio", snr_estimate=None):
        if isinstance(positions, int):
            t = torch.arange(positions, device=self.get_device()).float()
        else:
            t = positions.float().to(self.get_device())
        
        if not self.use_freq_bands:
            # Original implementation
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            freqs = freqs + self.bias[:freqs.shape[0]]
            
            if self.variable_radius:
                # Apply learnable radius instead of fixed radius=1
                radius = F.softplus(self.radius)  # Ensure radius is positive
                freqs = torch.polar(radius.unsqueeze(0).expand_as(freqs), freqs)
            else:
                # Original fixed radius
                freqs = torch.polar(torch.ones_like(freqs), freqs)
        else:
            # FrequencyBand implementation
            low = torch.einsum('i,j->ij', t, self.low_freq)
            mid = torch.einsum('i,j->ij', t, self.mid_freq)
            
            # Domain-specific high frequencies
            if domain == "audio":
                high = torch.einsum('i,j->ij', t, self.high_freq_audio)
            else:
                high = torch.einsum('i,j->ij', t, self.high_freq_text)
            
            # Apply bias
            if self.speech_enhanced:
                low = low + self.low_bias[:low.shape[0]]
                mid = mid + self.mid_bias[:mid.shape[0]]
                high = high + self.high_bias[:high.shape[0]]
            else:
                # Create full bias-adjusted frequencies before applying radius
                freqs = torch.cat([low, mid, high], dim=-1)
                freqs = freqs + self.bias[:freqs.shape[0]]
                low, mid, high = torch.split(freqs, freqs.shape[1]//3, dim=1)
            
            # Apply variable radius if enabled
            if self.variable_radius:
                # Get appropriate radius for each band
                low_radius = F.softplus(self.low_radius)
                mid_radius = F.softplus(self.mid_radius)
                
                if domain == "audio":
                    high_radius = F.softplus(self.high_radius_audio)
                else:
                    high_radius = F.softplus(self.high_radius_text)
                
                # Adjust radius based on SNR if provided (audio mode only)
                if snr_estimate is not None and domain == "audio":
                    # Convert SNR to a scaling factor (lower SNR = smaller high freq radius)
                    snr_factor = torch.sigmoid((snr_estimate - 5) / 5)  # Maps to 0-1
                    
                    # Apply progressively stronger scaling to higher frequencies
                    # (high frequencies most affected by noise)
                    low_radius = low_radius  # Low frequencies mostly preserved
                    mid_radius = mid_radius * (0.5 + 0.5 * snr_factor)  # Partial scaling
                    high_radius = high_radius * snr_factor  # Strongest scaling
                
                # Create complex numbers with variable radius for each band
                low_complex = torch.polar(low_radius.unsqueeze(0).expand_as(low), low)
                mid_complex = torch.polar(mid_radius.unsqueeze(0).expand_as(mid), mid)
                high_complex = torch.polar(high_radius.unsqueeze(0).expand_as(high), high)
                
                # Combine all bands
                freqs = torch.cat([low_complex, mid_complex, high_complex], dim=-1)
            else:
                # Use fixed radius=1 (original behavior)
                freqs = torch.cat([low, mid, high], dim=-1)
                freqs = torch.polar(torch.ones_like(freqs), freqs)
                
        return freqs
    
    def get_device(self):
        """Helper to get device from any parameter"""
        if hasattr(self, 'inv_freq'):
            return self.inv_freq.device
        return self.low_freq.device
        
    @staticmethod
    def apply_rotary(x, freqs):
        x1 = x[..., :freqs.shape[-1]*2]
        x2 = x[..., freqs.shape[-1]*2:]
        x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous() 
        x1 = torch.view_as_complex(x1)
        x1 = x1 * freqs
        x1 = torch.view_as_real(x1).flatten(-2)
        return torch.cat([x1.type_as(x), x2], dim=-1)
