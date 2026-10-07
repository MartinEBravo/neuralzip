# Third-party assets and licenses

Verified against upstream metadata on 7 October 2026. This inventory covers the
paper's evaluated models, source-bank donors, training checkpoints, optimizer
datasets, and principal software dependencies. Exact input revisions, hashes,
and donor assignments are retained in [the experimental catalogue](scripts/results/paper/catalogue.json);
this document does not change them.

License names below report the upstream declaration, not a new grant of rights
or a blanket assurance of redistribution permission. Follow the linked license,
acceptable-use policy, attribution, and access conditions for each asset.
Download model weights and datasets from their upstream providers; do not
assume that a code license also licenses a checkpoint or its training data.

## Bundled code and experiment baselines

| Component | License and authoritative source | Scope |
| --- | --- | --- |
| DFloat11 | Apache-2.0; [bundled license](third_party/dfloat11/LICENSE), [official source](https://github.com/LeanModels/DFloat11/blob/457733886ce6ebc6d8dda1621fad1ffa2661e028/LICENSE) | Adapted decoder components and the official experimental baseline. The experiment runner records the official source revision and file hashes. |
| Zstandard / Huff0 | BSD-3-Clause; [bundled license](third_party/zstd/LICENSE), [pinned upstream information](third_party/zstd/UPSTREAM) | Huff0 sources used by the native extension, revision `5233c58e6ca0b1c4c6b353ad79649191ed195bdc`. |
| ZipNN | MIT, as declared in [upstream LICENSE.txt](https://github.com/zipnn/zipnn/blob/main/LICENSE.txt) | Separately installed CPU baseline; upstream's attribution text is retained by that package. |

The existing bundled license files remain authoritative and unchanged.
No root license for NeuralZip's original code is currently declared in this
repository. This third-party inventory does not select one or relicense that code.

## Evaluated Hugging Face models and source-bank donors

Each model-card link is pinned to the measured input revision. For gated cards,
the license declaration was checked through the public revision-specific
`https://huggingface.co/api/models/<repository>/revision/<commit>` metadata,
without accessing credentials or downloading weights. Custom model licenses
are deliberately not described as MIT or Apache-2.0.

| Upstream asset / pinned model card | Input revision | Declared terms |
| --- | --- | --- |
| [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/README.md) | `2fc06364715b967f1860aea9cf38778875588b17` | Apache-2.0 |
| [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/README.md) | `15852e8c16360a2fea060d615a32b45270f8a8fc` | Apache-2.0 |
| [`google/gemma-3-1b-pt`](https://huggingface.co/google/gemma-3-1b-pt/blob/fcf18a2a879aab110ca39f8bffbccd5d49d8eb29/README.md) | `fcf18a2a879aab110ca39f8bffbccd5d49d8eb29` | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) |
| [`google/gemma-3-4b-pt`](https://huggingface.co/google/gemma-3-4b-pt/blob/cc012e0a6d0787b4adcc0fa2c4da74402494554d/README.md) | `cc012e0a6d0787b4adcc0fa2c4da74402494554d` | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) |
| [`facebook/dinov3-convnext-base-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-convnext-base-pretrain-lvd1689m/blob/e619358a5fe272554f17d2fb6cc65a679326bac0/README.md) | `e619358a5fe272554f17d2fb6cc65a679326bac0` | [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/) |
| [`facebook/dinov3-convnext-large-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-convnext-large-pretrain-lvd1689m/blob/e959efa74c867491dcfe3ec3e4f97382e39025b3/README.md) | `e959efa74c867491dcfe3ec3e4f97382e39025b3` | [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/) |
| [`facebook/dinov3-vitb16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m/blob/5931719e67bbdb9737e363e781fb0c67687896bc/README.md) | `5931719e67bbdb9737e363e781fb0c67687896bc` | [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/) |
| [`facebook/dinov3-vitl16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m/blob/ea8dc2863c51be0a264bab82070e3e8836b02d51/README.md) | `ea8dc2863c51be0a264bab82070e3e8836b02d51` | [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/) |
| [`google/siglip2-base-patch16-224`](https://huggingface.co/google/siglip2-base-patch16-224/blob/75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2/README.md) | `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` | Apache-2.0 |
| [`google/siglip2-large-patch16-256`](https://huggingface.co/google/siglip2-large-patch16-256/blob/787800c8990e6f058423089178e718139608408c/README.md) | `787800c8990e6f058423089178e718139608408c` | Apache-2.0 |
| [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/blob/541d1f99c6b0c3cd0b11a95167540bb8edefd82b/README.md) | `541d1f99c6b0c3cd0b11a95167540bb8edefd82b` | CC-BY-4.0 |
| [`nvidia/parakeet-tdt-0.6b-v2`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/blob/ae9ad07059c7c739ffaf932226a8fe64ae2620b0/README.md) | `ae9ad07059c7c739ffaf932226a8fe64ae2620b0` | CC-BY-4.0 |
| [`gpt2`](https://huggingface.co/gpt2/blob/607a30d783dfa663caf39e06633721c8d4cfcd7e/README.md) | `607a30d783dfa663caf39e06633721c8d4cfcd7e` | MIT |
| [`openai-community/gpt2-medium`](https://huggingface.co/openai-community/gpt2-medium/blob/6dcaa7a952f72f9298047fd5137cd6e4f05f41da/README.md) | `6dcaa7a952f72f9298047fd5137cd6e4f05f41da` | MIT |
| [`google-bert/bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased/blob/86b5e0934494bd15c9632b12f734a8a67f723594/README.md) | `86b5e0934494bd15c9632b12f734a8a67f723594` | Apache-2.0 |
| [`google-bert/bert-large-uncased`](https://huggingface.co/google-bert/bert-large-uncased/blob/6da4b6a26a1877e173fca3225479512db81a5e5b/README.md) | `6da4b6a26a1877e173fca3225479512db81a5e5b` | Apache-2.0 |
| [`Qwen/Qwen2.5-7B`](https://huggingface.co/Qwen/Qwen2.5-7B/blob/d149729398750b98c0af14eb82c78cfe92750796/README.md) | `d149729398750b98c0af14eb82c78cfe92750796` | Apache-2.0 |
| [`Qwen/Qwen2.5-14B`](https://huggingface.co/Qwen/Qwen2.5-14B/blob/97e1e76335b7017d8f67c08a19d103c0504298c9/README.md) | `97e1e76335b7017d8f67c08a19d103c0504298c9` | Apache-2.0 |
| [`Qwen/Qwen3.5-2B-Base`](https://huggingface.co/Qwen/Qwen3.5-2B-Base/blob/b1485b2fa6dfa1287294f269f5fb618e03d52d7c/README.md) | `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` | Apache-2.0 |
| [`Qwen/Qwen3.5-27B`](https://huggingface.co/Qwen/Qwen3.5-27B/blob/fc05daec18b0a78c049392ed2e771dde82bdf654/README.md) | `fc05daec18b0a78c049392ed2e771dde82bdf654` | Apache-2.0 |
| [`allenai/OLMo-2-1124-7B`](https://huggingface.co/allenai/OLMo-2-1124-7B/blob/7df9a82518afdecae4e8c026b27adccc8c1f0032/README.md) | `7df9a82518afdecae4e8c026b27adccc8c1f0032` | Apache-2.0 |
| [`allenai/OLMo-2-1124-13B`](https://huggingface.co/allenai/OLMo-2-1124-13B/blob/3fefddc1bf18a30e1d9b91000271630718f2aa8b/README.md) | `3fefddc1bf18a30e1d9b91000271630718f2aa8b` | Apache-2.0 |
| [`meta-llama/Llama-3.2-3B`](https://huggingface.co/meta-llama/Llama-3.2-3B/blob/13afe5124825b4f3751f836b40dafda64c1ed062/README.md) | `13afe5124825b4f3751f836b40dafda64c1ed062` | [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/) and Acceptable Use Policy |
| [`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B/blob/4e20de362430cd3b72f300e6b0f18e50e7166e08/README.md) | `4e20de362430cd3b72f300e6b0f18e50e7166e08` | [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/) and Acceptable Use Policy |
| [`microsoft/phi-4`](https://huggingface.co/microsoft/phi-4/blob/2db69c1c3e91a05d2c64a3185acfbaf36f744e25/README.md) | `2db69c1c3e91a05d2c64a3185acfbaf36f744e25` | MIT |
| [`google/gemma-3-12b-pt`](https://huggingface.co/google/gemma-3-12b-pt/blob/295efb63d01a7017928f273a94ebb86105c9526f/README.md) | `295efb63d01a7017928f273a94ebb86105c9526f` | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) |
| [`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/README.md) | `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` | CreativeML OpenRAIL-M |
| [`stabilityai/stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/462165984030d82259a11f4367a4eed129e94a7b/README.md) | `462165984030d82259a11f4367a4eed129e94a7b` | CreativeML Open RAIL++-M |
| [`deepseek-ai/deepseek-moe-16b-base`](https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/blob/521d2bc4fb69a3f3ae565310fcc3b65f97af2580/README.md) | `521d2bc4fb69a3f3ae565310fcc3b65f97af2580` | [DeepSeek Model License](https://github.com/deepseek-ai/DeepSeek-MoE/blob/main/LICENSE-MODEL) |
| [`deepseek-ai/deepseek-moe-16b-chat`](https://huggingface.co/deepseek-ai/deepseek-moe-16b-chat/blob/eefd8ac7e8dc90e095129fe1a537d5e236b2e57c/README.md) | `eefd8ac7e8dc90e095129fe1a537d5e236b2e57c` | [DeepSeek Model License](https://github.com/deepseek-ai/DeepSeek-MoE/blob/main/LICENSE-MODEL) |
| [`Qwen/Qwen3-30B-A3B`](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/README.md) | `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` | Apache-2.0 |
| [`Qwen/Qwen3-30B-A3B-Base`](https://huggingface.co/Qwen/Qwen3-30B-A3B-Base/blob/1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9/README.md) | `1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9` | Apache-2.0 |
| [`RZ412/Qwen2.5-7B-Instruct-S1K-DeepSeek-Distilled`](https://huggingface.co/RZ412/Qwen2.5-7B-Instruct-S1K-DeepSeek-Distilled/blob/4ecabac382c637b7103fc4d1a96e41c1712f60e4/README.md) | `4ecabac382c637b7103fc4d1a96e41c1712f60e4` | Apache-2.0 |
| [`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/blob/6a6f4aa4197940add57724a7707d069478df56b1/README.md) | `6a6f4aa4197940add57724a7707d069478df56b1` | MIT; the card also identifies the underlying [Llama 3.1 license](https://www.llama.com/llama3_1/license/) |
| [`nvidia/Llama-3.1-Nemotron-Nano-8B-v1`](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1/blob/54641c1611fcff44fa4865626462445e0a153fc7/README.md) | `54641c1611fcff44fa4865626462445e0a153fc7` | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) plus the card's Llama 3.1 terms |
| [`google/siglip2-so400m-patch16-384`](https://huggingface.co/google/siglip2-so400m-patch16-384/blob/dd658faac399427308559e2c3ac1e99cbe43845d/README.md) | `dd658faac399427308559e2c3ac1e99cbe43845d` | Apache-2.0 |
| [`google/siglip2-giant-opt-patch16-384`](https://huggingface.co/google/siglip2-giant-opt-patch16-384/blob/a713301b217d38485fb2204c808367d10bc3cc40/README.md) | `a713301b217d38485fb2204c808367d10bc3cc40` | Apache-2.0 |
| [`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo/blob/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/README.md) | `41f01f3fe87f28c78e2fbf8b568835947dd65ed9` | MIT |
| [`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3/blob/06f233fe06e710322aca913c1bc4249a0d71fce1/README.md) | `06f233fe06e710322aca913c1bc4249a0d71fce1` | Apache-2.0 in this pinned card; see discrepancy note below |

OpenAI's [Whisper repository](https://github.com/openai/whisper#license)
describes both code and weights as MIT, whereas the measured
`openai/whisper-large-v3` donor's pinned Hugging Face card declares
Apache-2.0. Both declarations are recorded rather than silently treating them
as identical. The turbo target's pinned card declares MIT.

The RZ412 distilled Qwen release is listed separately from its Qwen base model.
DeepSeek-R1-Distill-Llama and Nemotron retain the Llama-origin information stated
in their cards; an MIT or NVIDIA label does not erase those upstream terms.

## Training-checkpoint families

All 70 measured checkpoint commits are enumerated in the catalogue. The following
cards identify the families and their declared licenses; we did not retrain them.

| Checkpoint family / license source | Measured checkpoints | Declared terms |
| --- | --- | --- |
| [`almanach/Gaperon-1.5B-ckpts`](https://huggingface.co/almanach/Gaperon-1.5B-ckpts/blob/4c943cf5d7852149dd2805ba185754d4aca2e42c/README.md) | 8 pinned checkpoints in the catalogue | BigScience OpenRAIL-M |
| [`allenai/OLMo-2-0425-1B`](https://huggingface.co/allenai/OLMo-2-0425-1B/blob/a1847dff35000b4271fa70afc5db10fd29fedbdf/README.md) | 21 pinned checkpoints in the catalogue | Apache-2.0 |
| [`EleutherAI/pythia-1b`](https://huggingface.co/EleutherAI/pythia-1b/blob/f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2/README.md) | 21 pinned checkpoints in the catalogue | Apache-2.0 |
| [`trillionlabs/1.9B-Intermediate-Checkpoints`](https://huggingface.co/trillionlabs/1.9B-Intermediate-Checkpoints/blob/79a5c34ff5348f960882f29b587622f937c91ae8/README.md) | 20 pinned checkpoints in the catalogue | Apache-2.0 |

For Gaperon, OLMo-2-1B, and Pythia, the inspected final measured revision in each
family did not contain a README model card. Their license information is taken
from the same upstream repository's separately pinned release card:
Gaperon `4c943cf5d7852149dd2805ba185754d4aca2e42c`,
OLMo-2-1B `a1847dff35000b4271fa70afc5db10fd29fedbdf`,
and Pythia `f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2`.
This is a release-level declaration, not a claim that a license file was
embedded in every checkpoint commit.

## TorchVision pretrained weights

| Asset | Exact weight enum |
| --- | --- |
| `torchvision.models.resnet101` | `ResNet101_Weights.IMAGENET1K_V2` |
| `torchvision.models.resnet152` | `ResNet152_Weights.IMAGENET1K_V2` |
| `torchvision.models.convnext_base` | `ConvNeXt_Base_Weights.IMAGENET1K_V1` |
| `torchvision.models.convnext_large` | `ConvNeXt_Large_Weights.IMAGENET1K_V1` |
| `torchvision.models.efficientnet_b3` | `EfficientNet_B3_Weights.IMAGENET1K_V1` |
| `torchvision.models.efficientnet_b7` | `EfficientNet_B7_Weights.IMAGENET1K_V1` |

TorchVision code is [BSD-3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE).
Its [pretrained-model notice](https://github.com/pytorch/vision#pre-trained-model-license)
explicitly distinguishes model/data terms from the library license. We therefore
do not assign the code's BSD license to these ImageNet-trained weights.
See the relevant [model documentation](https://docs.pytorch.org/vision/stable/models.html)
and [ImageNet access terms](https://www.image-net.org/download.php).
The optimizer study initializes its ResNet-18 and ViT-B/16 architectures
randomly; it does not use pretrained weights for those training experiments.

## Optimizer-study datasets and tokenizer

The actual data loading, checksums, preprocessing and train/held-out splits are
specified in [the optimizer implementation](neuralzip/_experiment_optimizers.py).

| Asset and creator/source | License information |
| --- | --- |
| MNIST, LeCun, Cortes and Burges; obtained with TorchVision | The [curator-hosted card](https://huggingface.co/datasets/ylecun/mnist/blob/77f3279092a1c1579b2250db8eafed0ad422088c/README.md) reports MIT. The original LeCun site was not accessible during this check; this records that card's declaration rather than independently asserting terms for every mirror. |
| CIFAR-10, Krizhevsky, Nair and Hinton | The [creator's distribution page](https://www.cs.toronto.edu/~kriz/cifar.html) requests citation but gives no explicit dataset license. The [curator-hosted card](https://huggingface.co/datasets/uoft-cs/cifar10/blob/0b2714987fa478483af9968de7c934580d0bb9a2/README.md) labels the license `unknown`. Do not substitute TorchVision's code license. |
| CIFAR-100, Krizhevsky, Nair and Hinton | Same [creator distribution](https://www.cs.toronto.edu/~kriz/cifar.html); [curator-hosted card](https://huggingface.co/datasets/uoft-cs/cifar100/blob/aadb3af77e9048adbea6b47c21a81e47dd092ae5/README.md) labels the license `unknown`. |
| Tiny Shakespeare, Shakespeare texts assembled in Karpathy's char-rnn repository | The [source repository](https://github.com/karpathy/char-rnn#license) declares MIT; no separate dataset-specific license is supplied there. The runner's exact text SHA-256 is `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`. This is not a new license grant over underlying text. |
| GPT-2 tokenizer, OpenAI; `openai-community/gpt2` | MIT in the [pinned card](https://huggingface.co/openai-community/gpt2/blob/607a30d783dfa663caf39e06633721c8d4cfcd7e/README.md); same revision as the GPT-2 input above. |

Unknown or conflicting upstream declarations are disclosed, not resolved by
this inventory. These entries provide provenance and license information;
they do not authorize redistribution of datasets or weights.

## Principal external software

These are separately installed packages, not relicensed by NeuralZip.
Their bundled notices and exact installed-version licenses remain applicable.

| Software / source license | Use |
| --- | --- |
| [NumPy: BSD-3-Clause](https://github.com/numpy/numpy/blob/main/LICENSE.txt) | Array operations and statistics. |
| [PyTorch: BSD-style with additional bundled notices](https://github.com/pytorch/pytorch/blob/main/LICENSE) | Tensor operations, training, SGD, Adam, AdamW, and Muon implementation. |
| [TorchVision: BSD-3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE) | Architectures, transforms, and dataset loaders; not a dataset/weight license grant. |
| [Safetensors: Apache-2.0](https://github.com/huggingface/safetensors/blob/main/LICENSE) | Checkpoint I/O. |
| [Transformers: Apache-2.0](https://github.com/huggingface/transformers/blob/main/LICENSE) | Model/tokenizer implementations. |
| [Hugging Face Hub: Apache-2.0](https://github.com/huggingface/huggingface_hub/blob/main/LICENSE) | Pinned upstream downloads. |
| [Diffusers: Apache-2.0](https://github.com/huggingface/diffusers/blob/main/LICENSE) | Diffusion-model construction in recorded experiments. |
| [Accelerate: Apache-2.0](https://github.com/huggingface/accelerate/blob/main/LICENSE) | Model-loading support in recorded environments. |
| [NeMo: Apache-2.0](https://github.com/NVIDIA-NeMo/Speech/blob/main/LICENSE) | Parakeet model support in recorded environments. |
| [CuPy: MIT](https://github.com/cupy/cupy/blob/main/LICENSE) | CUDA kernels and DFloat11 runtime. |
| [SciPy: BSD-3-Clause](https://github.com/scipy/scipy/blob/main/LICENSE.txt) | Preliminary Jensen-Shannon calculations. |
| [nanobind: BSD-3-Clause](https://github.com/wjakob/nanobind/blob/master/LICENSE) | Native bindings. |
| [scikit-build-core: Apache-2.0](https://github.com/scikit-build/scikit-build-core/blob/main/LICENSE) | Source build backend. |
| [CMake: BSD-3-Clause](https://cmake.org/licensing/) | Native build configuration. |
| [NVIDIA CUDA Toolkit terms](https://docs.nvidia.com/cuda/eula/index.html) | Separately installed GPU toolchain/runtime; not covered by a model's or NeuralZip's license. |

Dependency declarations are in [pyproject.toml](pyproject.toml). Saved experimental
evidence under [scripts/results/](scripts/results/) records runtime versions and source hashes.
When reproducing a saved experiment, use its captured version and notices rather
than assuming today's upstream branch is the measured environment.
