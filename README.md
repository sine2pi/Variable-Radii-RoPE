tldr: The idea is to model relative distances between tokens and/or features along with the relative positions using standard RoPE.

This gives the model a more expressive way to represent "This token is at X position relative to that token and that positional relationship has Y importance" where both X and Y are learned during training.

Hopefully this captures more of the complex temporal dependencies in audio signals. It might be most useful in the encoder of a visual or audio tranformer model. I plopped it into my versitle but master of none rotary implimentation so it should be easy to test in any module without a lot of fuss.

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



# Rotary Embedding Configuration Options

There are a few configuration parameters that control different aspects of the rotary embeddings. Here's a comprehensive breakdown:

## Core Parameters

- `dims`: Embedding dimension (typically head_dim in attention)
- `ctx=1500`: Maximum context length
- `debug=False`: Whether to print debug information

## Advanced Embedding Control

- `variable_radius=False`: Use variable radius for complex numbers
- `learned_radius=False`: Make radius values trainable parameters
- `use_xpos=False`: Use xPos extension (position-dependent scaling)
- `xpos_scale_base=512`: Base for xPos scaling formula
- `interpolate_factor=1.0`: Scale positions for longer effective context

## Performance Options

- `cache_if_possible=True`: Cache computed frequencies for reuse
- `auto_detect_shape=True`: Automatically detect tensor shapes/formats

## Automatic Detection

Yes, when `auto_detect_shape=True` (the default), the implementation automatically:

1. Analyzes input tensor shapes to determine if it's:
   - Standard sequence format `[batch, seq_len, dim]`
   - Multi-head format `[batch, heads, seq_len, head_dim]`

2. Adapts the rotary calculations based on the detected format

3. Applies appropriate reshape operations for the tensor type

## Frequency Configuration

- `freqs_mode='lang'`: Mode for frequency generation
  - Options: `'lang'` (language), `'pixel'` (vision), `'constant'` (boring)
- `theta=10000`: Base for frequency computation in language mode
- `max_freq=10`: Maximum frequency when using pixel mode
- `learned_freq=False`: Whether frequencies are trainable

This rotary class works across different components (encoder layers, decoder layers, attention modules) without needing specific format adjustments.

## Integration Methods

The class provides several convenience methods:

In an attention module:

    rotary = Rotary(dims=head_dim)
    q, k = rotary.rotate_queries_and_keys(q, k)

 Directly at tensor level, without specifying context:
 
    x = rotary.apply_rotary(x) # Apply to any tensor (auto-detects format)

 Specialized for attention
 
    q, k = rotary.rotate_queries_and_keys(q, k)

 Full monty QKV handling
 
    q, k, v = rotary.rotate_qkv(q, k, v)

 For token embeddings
 
    embeddings = rotary.rotate_token_embeddings(embeddings)

 Or with a specific context length:
 
    x = rotary.apply_rotary(x, ctx=ctx)  # Shape auto-detected

```python
# Apply to any tensor (auto-detects format)
x = rotary.apply_rotary(x)

# Specialized for attention
q, k = rotary.rotate_queries_and_keys(q, k)

# Full QKV handling
q, k, v = rotary.rotate_qkv(q, k, v)

# For token embeddings
embeddings = rotary.rotate_token_embeddings(embeddings)
```

Each method handles the shape detection and appropriate transformations internally, making the implementation very flexible.




Basic working implimentation:

```python
    # # In an attention module:
    # rotary = Rotary(dims=head_dim)
    # q, k = rotary.rotate_queries_and_keys(q, k)

    # # Or directly at tensor level, without specifying context:
    # x = rotary.apply_rotary(x)  # Shape auto-detected

class Rotary(nn.Module):   
    def __init__( self, dims, ctx=1500, freqs_mode='lang', theta=10000, max_freq=10, learned_freq=False, variable_radius=False, learned_radius=False, use_xpos=False, xpos_scale_base=512, interpolate_factor=1.0, cache_if_possible=True, auto_detect_shape=True, debug=False ):
        super().__init__()
        self._counter = 0
        self.debug = debug
        self.dims = dims
        self.max_ctx = ctx
        self.variable_radius = variable_radius
        self.cache_if_possible = cache_if_possible
        self.use_xpos = use_xpos
        self.interpolate_factor = interpolate_factor
        self.auto_detect_shape = auto_detect_shape
        
        if freqs_mode == 'lang':
            freqs = 1.0 / (theta ** (torch.arange(0, dims, 2).float() / dims))
        elif freqs_mode == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dims // 2) * pi
        elif freqs_mode == 'constant':
            freqs = torch.ones(dims // 2).float()
        else:
            raise ValueError(f"Unknown freqs_mode: {freqs_mode}")
            
        self.inv_freq = nn.Parameter(freqs, requires_grad=learned_freq)
        self.freqs_mode = freqs_mode
        
        if variable_radius:
            self.radius = nn.Parameter(
                torch.ones(dims // 2),
                requires_grad=learned_radius
            )
            
        self.bias = nn.Parameter(torch.zeros(ctx, dims // 2))
        
        if use_xpos:
            scale = (torch.arange(0, dims, 2) + 0.4 * dims) / (1.4 * dims)
            self.scale_base = xpos_scale_base
            self.register_buffer('scale', scale, persistent=False)
            

        if use_xpos:
            self.register_buffer('cached_scales', torch.zeros(ctx, dims), persistent=False)
            self.cached_scales_ctx = 0
            
        self.register_buffer('dummy', torch.tensor(0), persistent=False)
        self.register_buffer('cached_freqs', torch.zeros(ctx, dims // 2), persistent=False)
        self.cached_freqs_ctx = 0
        
        self.ctx_cache = {}
        self.max_cache_entries = 16
        
        self.register_buffer('param_version', torch.tensor(0), persistent=False)
    
    def invalidate_cache(self):
        self.cached_freqs_ctx = 0
        self.ctx_cache.clear()
        self.param_version += 1

    @property
    def device(self):
        return self.dummy.device
    
    def get_seq_pos(self, ctx, device, dtype, offset=0):
        return (torch.arange(ctx, device=device, dtype=dtype) + offset) / self.interpolate_factor
    
    def get_scale(self, t, ctx=None, offset=0):
        assert self.use_xpos
        should_cache = (self.cache_if_possible and
            exists(ctx) and (offset + ctx) <= self.max_ctx)
        
        if (should_cache and exists(self.cached_scales) and
            (ctx + offset) <= self.cached_scales_ctx):
            return self.cached_scales[offset:(offset + ctx)]
            
        power = (t - len(t) // 2) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = repeat(scale, 'n d -> n (d r)', r=2)
        
        if should_cache and offset == 0:
            self.cached_scales[:ctx] = scale.detach()
            self.cached_scales_ctx = ctx
            
        return scale
    
    @autocast('cuda', enabled=False)
    def forward(self, x, offset=0):
        if isinstance(x, int):
            t = torch.arange(x, device=self.device, dtype=torch.float32)
            ctx = x
        else:
            t = x.float().to(self.device)
            ctx = t.shape[0] if hasattr(t, 'shape') else t
        
        skip_cache = self.inv_freq.requires_grad or (
            self.variable_radius and self.radius.requires_grad)
        
        if not skip_cache and self.cache_if_possible:
            if ctx <= self.max_ctx and offset == 0:
                if ctx <= self.cached_freqs_ctx:
                    return self.cached_freqs[:ctx].unsqueeze(0)
            
            cache_key = (ctx, offset, self.param_version.item())
            if cache_key in self.ctx_cache:
                return self.ctx_cache[cache_key]

        if self.freqs_mode == 'lang':
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        else:
            freqs = einsum('..., f -> ... f', t.type(self.inv_freq.dtype), self.inv_freq)
        freqs = freqs + self.bias[offset:offset+freqs.shape[0]]
        
        if self.variable_radius:
            radius = F.softplus(self.radius)
            freqs = torch.polar(radius.unsqueeze(0).expand_as(freqs), freqs)
        else:
            freqs = torch.polar(torch.ones_like(freqs), freqs)
        freqs = freqs.unsqueeze(0)
        
        if not skip_cache and self.cache_if_possible:
            if ctx <= self.max_ctx and offset == 0:
                self.cached_freqs[:ctx] = freqs.squeeze(0)
                self.cached_freqs_ctx = max(self.cached_freqs_ctx, ctx)
            else:
                if len(self.ctx_cache) >= self.max_cache_entries:
                    self.ctx_cache.pop(next(iter(self.ctx_cache)))
                self.ctx_cache[cache_key] = freqs.detach()
            
        if self.debug and self._counter < 1:
            print(f'ROTARY -- freqs: {freqs.shape}, t: {t.shape if hasattr(t, "shape") else None}')
            self._counter += 1
        return freqs
    
    def _reshape_for_multihead(self, freqs, head, head_dim=None):
        head_dim = head_dim or self.dims // head
        ctx = freqs.shape[1]
        complex_per_head = head_dim // 2
        
        if complex_per_head * head > freqs.shape[2]:
            freqs = freqs[:, :, :complex_per_head * head]
        elif complex_per_head * head < freqs.shape[2]:
            padding = torch.zeros(
                (freqs.shape[0], ctx, complex_per_head * head - freqs.shape[2]),
                device=freqs.device,
                dtype=freqs.dtype
            )
            freqs = torch.cat([freqs, padding], dim=2)
            
        return freqs.view(freqs.shape[0], ctx, head, complex_per_head)
    
    def _detect_tensor_format(self, tensor):
        shape = tensor.shape
        
        if len(shape) == 3:
            return {
                'format': 'sequence',
                'batch': shape[0],
                'ctx': shape[1], 
                'dim': shape[2],
                'is_multi_head': False
            }
        elif len(shape) == 4:
            return {
                'format': 'multihead',
                'batch': shape[0],
                'head': shape[1],
                'ctx': shape[2],
                'head_dim': shape[3],
                'is_multi_head': True
            }
        else:
            raise ValueError(f"Unsupported tensor shape: {shape}")
    
    def apply_rotary(self, x, freqs=None, seq_dim=-2, start_index=0, scale=1.0):
        """
        Args:
            x: Tensor to apply rotary embeddings to
            freqs: Optional pre-computed frequency tensor
            seq_dim: Dimension containing sequence positions
            start_index: Starting index for partial rotation
            scale: Scaling factor for rotations 
        """
        format_info = self._detect_tensor_format(x) if self.auto_detect_shape else None
        
        if format_info:
            if format_info['is_multi_head']:
                batch, head, ctx, head_dim = (format_info['batch'], format_info['head'], 
                    format_info['ctx'], format_info['head_dim'])
                
                if freqs is None:
                    freqs = self.forward(ctx)
                    freqs = self._reshape_for_multihead(freqs, head, head_dim)
                    freqs = freqs.permute(0, 2, 1, 3)
                
                x1 = x[..., :freqs.shape[-1]*2]
                x2 = x[..., freqs.shape[-1]*2:]
                x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous()
                x1 = torch.view_as_complex(x1)
                x1 = x1 * freqs * scale
                x1 = torch.view_as_real(x1).flatten(-2)
                return torch.cat([x1.type_as(x), x2], dim=-1)
            else:
                ctx = format_info['ctx']
                
                if freqs is None:
                    freqs = self.forward(ctx)
                
                return self._apply_rotary_to_sequence(x, freqs, start_index, scale)
        else:
            if freqs is None:
                if seq_dim < 0:
                    seq_dim = len(x.shape) + seq_dim
                ctx = x.shape[seq_dim]
                freqs = self.forward(ctx)
            
            return self._apply_rotary_to_sequence(x, freqs, start_index, scale)
    
    def _apply_rotary_to_sequence(self, x, freqs, start_index=0, scale=1.0):
        rot_dim = freqs.shape[-1] * 2
        end_index = start_index + rot_dim
        
        assert rot_dim <= x.shape[-1], f'Feature dimension {x.shape[-1]} is too small for rotation size {rot_dim}'
        
        t_left = x[..., :start_index]
        t_middle = x[..., start_index:end_index]
        t_right = x[..., end_index:]
        
        if torch.is_complex(freqs):
            cos = freqs.real
            sin = freqs.imag
        else:
            cos = torch.cos(freqs)
            sin = torch.sin(freqs)
        
        t_transformed = (t_middle * cos * scale) + (rotate_half(t_middle) * sin * scale)
        return torch.cat((t_left, t_transformed, t_right), dim=-1)
    
    def rotate_queries_and_keys(self, q, k, scale=1.0):
        q_info = self._detect_tensor_format(q) if self.auto_detect_shape else None
        k_info = self._detect_tensor_format(k) if self.auto_detect_shape else None
        q_len = q_info['ctx'] if q_info else q.shape[-2]
        k_len = k_info['ctx'] if k_info else k.shape[-2]
        q_freqs = self.forward(q_len)
        
        if q_len != k_len:
            k_freqs = self.forward(k_len)
        else:
            k_freqs = q_freqs
            
        if self.use_xpos:
            q_seq = self.get_seq_pos(q_len, device=q.device, dtype=q.dtype)
            k_seq = self.get_seq_pos(k_len, device=k.device, dtype=k.dtype)
            q_scale = self.get_scale(q_seq).to(q.dtype) * scale
            k_scale = self.get_scale(k_seq).to(k.dtype) ** -1
        else:
            q_scale = scale
            k_scale = 1.0
        
        rotated_q = self.apply_rotary(q, q_freqs, scale=q_scale)
        rotated_k = self.apply_rotary(k, k_freqs, scale=k_scale)
        return rotated_q, rotated_k
    
    def rotate_qkv(self, q, k, v, scale=1.0):
        rotated_q, rotated_k = self.rotate_queries_and_keys(q, k, scale)
        return rotated_q, rotated_k, v
    
    def rotate_token_embeddings(self, token_emb, scale=1.0):
        ctx = token_emb.shape[1]
        freqs = self.forward(ctx)
        return self.apply_rotary(token_emb, freqs, scale=scale)
    

    

```


