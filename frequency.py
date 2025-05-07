class Rotary(nn.Module):
    def __init__(self, dims, max_ctx=1500, learned_freq=True, use_freq_bands=False, speech_enhanced=False):
        super().__init__()
        self.dims = dims
        self.use_freq_bands = use_freq_bands
        
        if not use_freq_bands:
            # Original implementation
            self.inv_freq = nn.Parameter(
                1.0 / (10000 ** (torch.arange(0, dims, 2) / dims)),
                requires_grad=learned_freq
            )
            self.bias = nn.Parameter(torch.zeros(max_ctx, dims // 2))
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
            
            # Text-specific high frequencies (optional differentiation)
            self.high_freq_text = nn.Parameter(
                1.0 / (10000 ** (torch.arange(2*band_size, 3*band_size, 2) / dims)),
                requires_grad=learned_freq
            )
            
            # Bias terms
            if speech_enhanced:
                # Separate bias for each frequency band
                self.low_bias = nn.Parameter(torch.zeros(max_ctx, band_size // 2))
                self.mid_bias = nn.Parameter(torch.zeros(max_ctx, band_size // 2))
                self.high_bias = nn.Parameter(torch.zeros(max_ctx, band_size // 2))
            else:
                # Single bias for all bands
                self.bias = nn.Parameter(torch.zeros(max_ctx, dims // 2))
                
        self.speech_enhanced = speech_enhanced and use_freq_bands

    def forward(self, positions, domain="audio"):
        if isinstance(positions, int):
            t = torch.arange(positions, device=self.get_device()).float()
        else:
            t = positions.float().to(self.get_device())
            
        if not self.use_freq_bands:
            # Original implementation
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            freqs = freqs + self.bias[:freqs.shape[0]]
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
                freqs = torch.cat([low, mid, high], dim=-1)
            else:
                freqs = torch.cat([low, mid, high], dim=-1)
                freqs = freqs + self.bias[:freqs.shape[0]]
                
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
