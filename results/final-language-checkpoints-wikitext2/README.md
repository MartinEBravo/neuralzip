# Final language checkpoint trajectories

GPT-2 and BERT Base were trained from random initialization for 20 epochs on
WikiText-2.  The exact FP32 initialization and every epoch checkpoint were
compressed on `relela-06` with 12 CPU threads.  NeuralZip uses one structural
plan frozen at epoch 0; ZipNN source rebuilds its local codes for each input.

`checkpoint_results.csv` contains one bitwise-verified row per model, epoch,
and compressor.  Setup and target mapping are reported separately from direct
encoding time.  The ratio covers exponent symbols, raw fallbacks, and metadata.
