

# Variable Radius in RoPE (Rotary Position Embeddings)

The `variable_radius` flag in this Rotary implementation adds a learnable magnitude to the complex numbers used in RoPE. This is a significant extension to the standard RoPE approach, which traditionally uses only unit-radius complex numbers (magnitude = 1).

## How it works:

1. **Parameter Creation:**
   ```python
   if variable_radius:
       self.radius = nn.Parameter(
           torch.ones(dims // 2),
           requires_grad=learned_radius
       )
   ```
   This creates a separate learnable radius parameter for each pair of dimensions. The model initializes with radius=1 (like standard RoPE) but can learn different values during training.

2. **Radius Calculation:**
   ```python
   if self.variable_radius:
       radius = F.softplus(self.radius)
       freqs = torch.polar(radius.unsqueeze(0).expand_as(freqs), freqs)
   else:
       freqs = torch.polar(torch.ones_like(freqs), freqs)
   ```

   Key operations:
   - `F.softplus(self.radius)`: Applies softplus activation to ensure all radii remain positive
   - `radius.unsqueeze(0).expand_as(freqs)`: Expands the radius values to match the shape of position frequencies
   - `torch.polar(radius, freqs)`: Creates complex numbers with the learned magnitudes (radii) and calculated phases (freqs)

3. **Effect on Complex Rotation:**
   When these complex numbers with variable radii are used in `apply_rotary()`, the complex multiplication `x1 * freqs` now includes both:
   - Rotation (from the phase angles)
   - Scaling (from the learned radii)

## Why This Matters

This approach gives the model more flexibility by allowing it to learn:
- Which frequency components to emphasize (larger radius)
- Which to diminish (smaller radius)
- Different scaling for different dimensions

Standard RoPE can only rotate embeddings, while this variable radius extension can both rotate AND scale them. This provides a richer way to encode positional information, potentially improving the model's ability to handle sequences of various lengths and capture position-dependent patterns.

Basic working implimentation:

```python

# usage:
#
# in attention module init:
#         self.rotary = Rotary(dims=dims, max_ctx=1500, learned_freq=True,  variable_radius=True)
# in forward:
#        self.freq = self.rotary(ctx)
#        q = self.rotary.apply_rotary(q, self.freq)
#        k = self.rotary.apply_rotary(k, self.freq)

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
    
    @staticmethod
    def apply_rotary(x, freqs):
        x1 = x[..., :freqs.shape[-1]*2]
        x2 = x[..., freqs.shape[-1]*2:]
        x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous() 
        x1 = torch.view_as_complex(x1)
        x1 = x1 * freqs
        x1 = torch.view_as_real(x1).flatten(-2)
        return torch.cat([x1.type_as(x), x2], dim=-1)
```

1. **Unified Interface**: Combines frequency band support and variable radius in one class
2. **SNR-Aware Radius Adjustment**: For audio processing, includes SNR-based scaling of radius (when provided)
3. **Per-Band Radius Control**: Different frequency bands can have different radius values
4. **Full Backward Compatibility**: Works with original settings while allowing gradual adoption
5. **Domain-Specific Handling**: Different parameters for audio vs text frequencies
