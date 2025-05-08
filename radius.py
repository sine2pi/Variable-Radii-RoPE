
class Rotary(nn.Module):
    def __init__(self, dims, max_ctx=1500, learned_freq=True, variable_radius=True, learned_radius=True):
        super().__init__()
        self.dims = dims
        self.variable_radius = variable_radius
        
        self.inv_freq = nn.Parameter(
            1.0 / (10000 ** (torch.arange(0, dims, 2) / dims)),
            requires_grad=learned_freq
        )
        
        if variable_radius:
            self.radius = nn.Parameter(
                torch.ones(dims // 2),
                requires_grad=learned_radius
            )
        
        self.bias = nn.Parameter(torch.zeros(max_ctx, dims // 2))
        
    def forward(self, positions):
        if isinstance(positions, int):
            t = torch.arange(positions, device=self.inv_freq.device).float()
        else:
            t = positions.float().to(self.inv_freq.device)
            
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        freqs = freqs + self.bias[:freqs.shape[0]]
        
        if self.variable_radius:
            radius = F.softplus(self.radius)
            freqs = torch.polar(radius.unsqueeze(0).expand_as(freqs), freqs)
        else:
            freqs = torch.polar(torch.ones_like(freqs), freqs)
            
        return freqs
    
    def _reshape_for_multihead(self, freqs, head, head_dim):
        ctx = freqs.shape[0]
        complex_per_head = head_dim // 2
        if complex_per_head * head > freqs.shape[1]:
            freqs = freqs[:, :complex_per_head * head]
        elif complex_per_head * head < freqs.shape[1]:
            padding = torch.zeros(
                (ctx, complex_per_head * head - freqs.shape[1]), 
                device=freqs.device, 
                dtype=freqs.dtype
            )
            freqs = torch.cat([freqs, padding], dim=1)
        freqs = freqs.view(ctx, head, complex_per_head)
        return freqs.permute(2, 1, 0, 2).unsqueeze(0)

    @staticmethod
    def apply_rotary(x, freqs):
        multihead_format = len(freqs.shape) == 4
        
        if multihead_format:
            x1 = x[..., :freqs.shape[-1]*2]
            x2 = x[..., freqs.shape[-1]*2:]
            
            x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous()
            x1 = torch.view_as_complex(x1)
            
            x1 = x1 * freqs
            
            x1 = torch.view_as_real(x1).flatten(-2)
            return torch.cat([x1.type_as(x), x2], dim=-1)
        else:
            x1 = x[..., :freqs.shape[-1]*2]
            x2 = x[..., freqs.shape[-1]*2:]
            x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous() 
            x1 = torch.view_as_complex(x1)
            x1 = x1 * freqs
            x1 = torch.view_as_real(x1).flatten(-2)
            return torch.cat([x1.type_as(x), x2], dim=-1)
