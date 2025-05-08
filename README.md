

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


1. **Unified Interface**: Combines frequency band support and variable radius in one class
2. **SNR-Aware Radius Adjustment**: For audio processing, includes SNR-based scaling of radius (when provided)
3. **Per-Band Radius Control**: Different frequency bands can have different radius values
4. **Full Backward Compatibility**: Works with original settings while allowing gradual adoption
5. **Domain-Specific Handling**: Different parameters for audio vs text frequencies
