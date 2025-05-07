class Rotary(nn.Module):
    def __init__(self, dims, max_ctx=1500, learned_freq=True, variable_radius=False, learned_radius=True):
        super().__init__()
        self.dims = dims
        self.variable_radius = variable_radius
        
        # Frequency parameters
        self.inv_freq = nn.Parameter(
            1.0 / (10000 ** (torch.arange(0, dims, 2) / dims)),
            requires_grad=learned_freq
        )
        
        # Optional radius parameters
        if variable_radius:
            self.radius = nn.Parameter(
                torch.ones(dims // 2),
                requires_grad=learned_radius
            )
        
        # Bias parameters
        self.bias = nn.Parameter(torch.zeros(max_ctx, dims // 2))
        
    def forward(self, positions):
        if isinstance(positions, int):
            t = torch.arange(positions, device=self.inv_freq.device).float()
        else:
            t = positions.float().to(self.inv_freq.device)
            
        # Calculate angles
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        freqs = freqs + self.bias[:freqs.shape[0]]
        
        # Apply radius
        if self.variable_radius:
            # Use learnable radius
            radius = F.softplus(self.radius)  # Ensure positive
            freqs = torch.polar(radius.unsqueeze(0).expand_as(freqs), freqs)
        else:
            # Use fixed radius=1 (original behavior)
            freqs = torch.polar(torch.ones_like(freqs), freqs)
            
        return freqs
    
    @staticmethod
    def apply_rotary(x, freqs):
        x1 = x[..., :freqs.shape[-1]*2]
        x2 = x[..., freqs.shape[-1]*2:]
        x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous() 
        x1 = torch.view_as_complex(x1)
        x1 = x1 * freqs
        x1 = torch.view_as_real(x1).flatten(-2)
        return torch.cat([x1.type_as(x), x2], dim=-1)
