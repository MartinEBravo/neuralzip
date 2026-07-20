# Final CIFAR-10 federated run

Final thesis artifacts produced on `relela-06` on 2026-07-17/18 with
`examples/run_final_federated_suite.py`.

- Dataset: real CIFAR-10 from the Python archive.
- Protocol: ResNet-18 and ResNet-34, 200 communication rounds, 10 clients,
  one local batch per client and round, batch size 64, seed 2026, FP32.
- Compression: ZipNN source and NeuralZip, 12 pinned CPU threads and one
  verified warm-up. The thesis reports `downlink_encode_seconds`: one direct
  encode of the global model per round. Client encoding, training and decoding
  are excluded from this timing. The CSV also retains the downlink-plus-slowest-
  uplink critical-path field as a diagnostic.
- Ratio: exact exponent symbols plus raw fallback and metadata. The final
  NeuralZip ratio charges the one-time codebook payload once per client.
- Trajectory: the NeuralZip-decoded state is used only after bitwise equality
  with the ZipNN-source decoded state has been verified.
- Completeness: each architecture contains 400 rows = 200 rounds x 2
  compressors. All round and summary rows passed `roundtrip_ok`.

Each architecture directory contains its complete per-round CSV, aggregate
summary and run metadata. The real CIFAR-10 archive remains on the experiment
host and is intentionally omitted from Git.
