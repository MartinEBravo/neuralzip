# Four-model training trajectories

This directory combines the final GPT-2 and BERT Base WikiText-2 trajectories
with the ResNet-18 and ResNet-34 vision trajectories. The checkpoint figure
uses CIFAR-100 for the ResNets; the federated figure uses CIFAR-10.

`examples/render_four_model_training_suite.py` validates complete compressor
pairs before writing either canonical CSV or 2x4 figure. The checkpoint figure
shows direct encoding time. The federated figure shows one direct downlink
encode per round. Its ratio panels share 0.1-percentage-point tick spacing so
the small size trade-off is not visually exaggerated.
All four federated source trajectories were executed sequentially on
`relela-06` before this combined artifact was regenerated.
The BERT Base and ResNet-18 inputs are GC-clean reruns: automatic cyclic
garbage collection is excluded from each timed encode for both compressors.
GPT-2 and ResNet-34 retain their earlier sequential trajectories, which did not
show periodic NeuralZip GC events.
