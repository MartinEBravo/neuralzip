# Four-model training trajectories

This directory combines the final GPT-2 and BERT Base WikiText-2 trajectories
with the ResNet-18 and ResNet-34 vision trajectories. The checkpoint figure
uses CIFAR-100 for the ResNets; the federated figure uses CIFAR-10.

`examples/render_four_model_training_suite.py` validates complete compressor
pairs before writing either canonical CSV or 2x4 figure. The checkpoint figure
shows direct encoding time. The federated figure shows cumulative paired
encoding speedup because the language models trained concurrently on separate
GPUs while sharing host load. This paired metric is empirically less sensitive
to that load transition; raw measured times remain in the source CSVs.
