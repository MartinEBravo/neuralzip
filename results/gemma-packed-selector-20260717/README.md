# Gemma packed precompute probe

Target-only probes for Gemma 3 1B on Godel, using 12 CPU threads and commit
`ca71949`. Both runs evaluate `n = {2, 3, 4}`, `top_k = 254`, a 30% packed
source budget, and a 1.0x packed extra-time budget.

The JS-cluster plan evaluated 546 packed modes across 182 exponent clusters.
None beat the selected local Huff0 base, so planned packed coverage remained
0%. Precompute took 21.71 s.

The tensor-granularity diagnostic evaluated 1,020 modes across 340 tensors. It
found two `n=2` opportunities:

- `layers.9.self_attn.o_proj.weight`
- `layers.25.self_attn.q_proj.weight`

Together they cover 2,359,296 exponents, or 0.236% of the model. The setup
estimated a 225-byte saving. A targeted encode of only these tensors confirmed
a 229-byte net saving after 2,345 bytes of referenced codebook metadata, with
bitwise round-trip for both compressors. This is a 0.009706 percentage-point
gain on the selected tensors but only 0.0000229 percentage points when
amortized over the complete Gemma exponent field. Tensor-granularity
precompute took 69.78 s, 3.21x the JS-cluster setup.

The result shows that cluster-level selection can hide isolated packed
opportunities, but their total benefit in Gemma is too small to justify a
production per-tensor packed plan under the current Top-254 representation.
