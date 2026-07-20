# Final WikiText-2 federated run

Final thesis artifacts produced on `relela-06` on 2026-07-19 with
`examples/run_final_federated_language_suite.py`.

- Dataset: real WikiText-2, reconstructed into 629 articles and assigned as
  whole documents to 10 clients with byte-balanced partitions.
- Models: randomly initialized GPT-2 Small for causal language modeling and
  BERT Base for 15% masked language modeling. Only the pretrained tokenizers
  are reused.
- Protocol: 200 communication rounds, 10 clients and one local batch of 2,048
  tokens per client and round. GPT-2 uses 8 sequences of length 256; BERT uses
  16 sequences of length 128.
- Compression: ZipNN source and NeuralZip with 12 CPU threads. The critical
  time is the downlink encode plus the slowest client uplink encode; training,
  decoding and transfer are excluded.
- Pairing: both compressors receive identical canonical FP32 storages in every
  round, tied aliases are transmitted once, and the trajectory advances only
  after bitwise-equal decoded states are verified.
- Setup accounting: `setup_bytes_once` contains serialized shared-codebook
  metadata. It does not serialize or claim the complete structural map.
- Reporting: the two language trajectories trained concurrently on separate
  A5000 GPUs and therefore shared host load. Raw paired timings are retained in
  the CSVs; the trajectory figure reports their cumulative within-round
  speedup. Across the observed host-load transition, median absolute times
  change by 8--10%, whereas the within-segment ratio of median times changes by
  at most 1.5%; the figure therefore uses the less load-sensitive paired
  comparison. Exponent ratios depend only on measured payload bytes and are
  unaffected by host load.
- Completeness: each model contains 400 rows = 200 rounds x 2 compressors. All
  rows and summaries pass `roundtrip_ok`.

Each model directory contains its complete per-round CSV, aggregate summary
and run metadata. Resume checkpoints and the WikiText-2 artifact remain on the
experiment host and are intentionally omitted from Git.
