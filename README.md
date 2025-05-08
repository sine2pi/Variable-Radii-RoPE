tldr: The idea is to model relative distances between tokens and/or features along with the relative positions using standard RoPE.

This gives the model a more expressive way to represent "This token is at X position relative to that token and that positional relationship has Y importance" where both X and Y are learned during training.

Hopefully this captures more of the complex temporal dependencies in audio signals. It might be most useful in the encoder of a visual or audio tranformer model.

Standard RoPE rotates embeddings along a 2d 1-unit circle on a plane. This variable radius extension is an attempt at adding scale. This might provide a richer way to encode positional information, potentially improving the model's ability to handle sequences of various lengths and capture position-dependent patterns. Probably good for sound. Thats the intuition anyway.

Next step is adding real audio frequency information to the nmess.. mockup is in the same folder.

### Particularly Good for Speech?

1. **Time-scale invariance**: Speech has natural elasticity (people speak at different speeds) - variable radius allows the model to learn which temporal distances matter most regardless of absolute speed

2. **Frequency-selective attention**: Some frequency components of speech carry more information than others (like formant transitions) - learned radii can emphasize the most relevant positional frequencies

3. **Context-dependent relationships**: In speech, the importance of a relationship between positions varies:
   - Phoneme boundaries need strong positional signals
   - Steady vowel states need less precise positioning
   - Silence regions need minimal positional emphasis

4. **Distance-dependent weighting**: This allows the model to learn how much "distance matters" for different frequency components, which is crucial as some speech patterns are local while others span longer contexts

- Rotation (angle) for encoding relative positions
- Scaling (radius) for encoding positional importance


The `variable_radius` flag in this Rotary implementation adds a learnable magnitude to the complex numbers used in RoPE. This is a significant extension to the standard RoPE approach, which traditionally uses only unit-radius complex numbers (magnitude = 1).

This is an attempt at making rope more interesting and useful for asr encoders or any data that might benefit from any radius other than 1.


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

# use the same way in your encoder or decoder.

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

```


