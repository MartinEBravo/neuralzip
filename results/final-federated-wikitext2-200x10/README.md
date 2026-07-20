# Final WikiText-2 federated run

Final thesis artifacts produced sequentially on `relela-06` on 2026-07-19/20 with
`examples/run_final_federated_language_suite.py`.

- Dataset: real WikiText-2, reconstructed into 629 articles and assigned as
  whole documents to 10 clients with byte-balanced partitions.
- Models: randomly initialized GPT-2 Small for causal language modeling and
  BERT Base for 15% masked language modeling. Only the pretrained tokenizers
  are reused.
- Protocol: 200 communication rounds, 10 clients and one local batch of 2,048
  tokens per client and round. GPT-2 uses 8 sequences of length 256; BERT uses
  16 sequences of length 128.
- Compression: ZipNN source and NeuralZip with 12 CPU threads. The reported
  encoding time is `downlink_encode_seconds`: one direct encode of the global
  model per round. Client encoding, decoding, training and transfer are
  excluded from this timing. The exponent ratio aggregates the downlink and all
  10 client uplinks in each round.
- Pairing: both compressors receive identical canonical FP32 storages in every
  round, tied aliases are transmitted once, and the trajectory advances only
  after bitwise-equal decoded states are verified.
- Setup accounting: `setup_bytes_once` contains serialized shared-codebook
  metadata. It does not serialize or claim the complete structural map.
- Reporting: GPT-2 completed before BERT began, so the two trajectories did
  not share host load with another compression process. The trajectory figure
  retains every directly measured downlink time. Exponent ratios depend only
  on measured payload bytes.
- Timing stability: BERT Base is the GC-clean rerun. Automatic cyclic garbage
  collection is suspended inside each timed encode and restored immediately
  afterward for both compressors. GPT-2 retains its earlier sequential run,
  which did not exhibit periodic NeuralZip GC events.
- Completeness: each model contains 400 rows = 200 rounds x 2 compressors. All
  rows and summaries pass `roundtrip_ok`.

Each model directory contains its complete per-round CSV, aggregate summary
and run metadata. Resume checkpoints and the WikiText-2 artifact remain on the
experiment host and are intentionally omitted from Git.
