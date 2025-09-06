
Another variation on my (old by now) variable radii rotary.

original 2022:

Conditional radius: radius = 1.0 + xa[:, :, :n.head_dim // 2]
This is the key insight for modulating the vector's magnitude.
The standard RoPE uses a constant radius (often implicitly 1.0).
This makes the radius a function of the pitch (xa). By adding 1.0 + xa, we ensure that the radius is never zero and that a pitch of zero doesn't destroy the positional information.
This gives the model an additional dimension of information. A token with a high pitch would have a larger magnitude, while a lower pitch would result in a smaller magnitude. This can be interpreted by the model as a stronger or weaker "signal" from that token.

Complex polar representation: freqs = torch.polar(radius, freqs)
The model can then perform its matrix multiplications using these complex numbers, allowing it to interpret both the position (via the angle) and the pitch (via the radius) simultaneously. 
