# Fair ZipNN local/source timing (2026-07-17)

The measurements were executed on `godel` using ZipNN 0.5.4, BF16 weights,
12 threads, CPU affinity `taskset -c 0-11`, one complete warm-up, and one
measured repetition. Every row passed a bitwise-exact round trip.

The official ZipNN compressor mutates its input tensor. Its non-destructive
model-compression path therefore prepares a private tensor copy. The artifacts
report the copy and codec wall times separately:

- `codec_encode_seconds`: compression call only, with the private input already
  prepared.
- `input_copy_seconds`: time required to preserve the original model tensor.
- `encode_seconds`: non-destructive encoding time, equal to codec plus copy for
  `zipnn-src` and to codec time for the read-only local implementation.

Both implementations receive the same 12-thread, 0.95 threshold, and 256 KiB
chunk configuration. The four per-model CSV/JSON pairs are the authoritative
raw outputs copied from `/tmp/neuralzip-zipnn-fair-20260717/outputs` on Godel.
