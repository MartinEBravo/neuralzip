# Experimental evidence and reproduction

Run commands from the repository root. The five public scripts write new runs
under `results/reruns/` and refuse to overwrite existing measurements. Model
downloads and temporary BF16 inputs go under `.cache/experiments/` by default.
Both directories are ignored by Git. No weights or setup pickles are distributed.

## Saved results: no training, downloads or timing

`paper/catalogue.json` fixes model/donor revisions, complete BF16 tensor
manifests, checkpoint steps and transfer-gate versions. `paper/evidence.zip`
contains CPU samples, GPU trials, entropy populations and optimizer traces.
`paper/evidence-members.json` records each distributed member's SHA256 and,
where available, its original pre-anonymization SHA256. Only identifying paths
and host names were normalized; numerical values and tensor hashes were retained.
The earlier GPU acceptance archives in `packed-gpu/` are unchanged.

```bash
python scripts/six_models_ablations.py --check
python scripts/all_models.py --check
python scripts/training_checkpoints.py --check
python scripts/preliminaries.py --check
python scripts/optimizers.py --check
```

Replace `--check` with `--export-saved` to generate CSVs from verified records.
Choose a new `--output` directory for each export. This reproduces recorded
numerical tables/plot coordinates, not fresh timings. Table 3 evidence contains
18 CPU cohorts and 15 complete same-device GPU cohorts; the checker explicitly
reports the other three as absent in this snapshot, not as measured or failed.
Six-suite GPU comparisons and isolated sharing/packing controls are retained in
the [GPU report](packed-gpu/RESULTS.md) and its raw archives.

## Captured software environment

The compression/optimizer spot-checks used Linux x86-64, Python 3.11,
PyTorch `2.9.1+cu128`, torchvision `0.24.1+cu128`, Transformers `5.12.1`,
NumPy `2.4.4`, SciPy `1.17.1`, CuPy `13.6.0`, ZipNN `0.5.4`,
Hugging Face Hub `1.29.0`, safetensors `0.8.0` and tokenizers `0.22.2`.
GPU spot-checks used an RTX A5500. Saved records retain their actual device and
protocol: GB10 and A5500 latencies must not be compared as the same hardware.

Use a separate environment for the archived experiments. The public HF wrapper
requires a newer Transformers release in `pyproject.toml`; installing its normal
dependencies would replace the archived experiment version. One setup is:

```bash
python3.11 -m venv .venv-experiments
source .venv-experiments/bin/activate
python -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install transformers==5.12.1 numpy==2.4.4 scipy==1.17.1 \
  cupy-cuda12x==13.6.0 zipnn==0.5.4 huggingface-hub==1.29.0 \
  safetensors==0.8.0 tokenizers==0.22.2 dahuffman==0.4.2 accelerate
python -m pip install --no-deps -e .
```

Stable Diffusion GPU construction additionally needs the `diffusers` version
recorded in its saved GPU environment. Access to gated checkpoints must already
be configured in the local Hugging Face client; scripts never copy credentials.
The scripts download pinned checkpoints if inputs are absent. `--inputs` can
instead point to trusted server-local `.pt` dictionaries named by model key;
every tensor's name, shape, dtype and bits are checked against the catalogue.

DFloat11 is optional for NeuralZip/ZipNN CPU runs. To enable its comparisons,
use the official source at the recorded revision:

```bash
git clone https://github.com/LeanModels/DFloat11 .cache/DFloat11
git -C .cache/DFloat11 checkout 457733886ce6ebc6d8dda1621fad1ffa2661e028
export PYTHONPATH="$PWD:$PWD/.cache/DFloat11:$PWD/.cache/DFloat11/dfloat11"
```

The GPU benchmark checks official Python, encoder, CUDA and PTX hashes. CPU
DFloat11 imports only the pinned encoder utility, not its CUDA package initializer.
Both adapters preserve the official codec; singleton/LUT fallbacks are reported.

## Fresh compression and ablation runs

Fresh runs use the current CPU codec, 256 KiB chunks, JS distance 0.05, adaptive
packed/singleton coding and size-only selection. Transfer uses the measured
source-only physical codebook bank and 3-bit drift gate: no target fitting,
and target routing is charged inside encoding. Own setup is measured once,
separately. CPU runs use 12 threads on cores 0–11, four warmups, and respectively
20/9/5 repetitions for the six-suite/Table 3/checkpoints. Change `--cpus` only
when the machine does not offer those cores, and report the change.

```bash
# One six-suite control, including the original same-call setup-phase profiler.
python scripts/six_models_ablations.py --models dinov3-vit --methods new \
  --verify-published --output results/reruns/six-vit

# One Table 3 target with its declared source-only donor.
python scripts/all_models.py --models resnet101 --methods new transfer zipnn \
  --verify-published --output results/reruns/resnet

# One original training step, not a rerun of all 70 checkpoints.
python scripts/training_checkpoints.py --families pythia-1b --steps 143000 \
  --verify-published --output results/reruns/pythia-final
```

`--verify-published` requires identical input identity, compressed sizes and
archive SHA256; it does not require elapsed times to be identical. For a short
spot-check, add `--repetitions 1` (the four warmups remain). Do not present that
as a full timing replication.

Six-suite `--methods` also accepts `sharing`, `packing`, `level-global`,
`level-tensor` and the `js-*` values shown by `--help`. Strict global means one
model-wide selected table, not hundreds of duplicated prepared tables; strict
tensor means one selected table per tensor. Phase timings, packing decisions
and physical archive/table metadata are retained in JSON, not just a summary CSV.
`--methods dfloat11` runs the original single tensorwise CPU encoding comparison,
including frequency/LUT/restart construction, with roundtrip checks outside timing.

Figure 6 intentionally retains the gate version recorded for each original
point. Most points use the 4-bit bank; the separately approved late-OLMo fixes
use the 3-bit gate. Reproduction follows those recorded versions rather than
silently replacing every point with a different setup. Figure 6 exports only
the current NeuralZip and ZipNN curves, not the removed DFloat11 curve.

## Fresh GPU inference

Prepare `new transfer sharing packing` CPU setups first for a six-suite model;
Table 3 needs `new transfer`. Keep these frozen files on the GPU host. Use a
new output directory, one physical GPU and one fresh process per method:

```bash
python scripts/six_models_ablations.py --models dinov3-vit --device gpu \
  --methods bf16 --gpu 0 --setups results/reruns/six-vit \
  --output results/reruns/gpu-vit
python scripts/six_models_ablations.py --models dinov3-vit --device gpu \
  --methods own --gpu 0 --setups results/reruns/six-vit \
  --output results/reruns/gpu-vit
```

Run BF16 first; other methods require its same-campaign, bitwise output reference.
Then use `transfer`, `dfloat11`, `sharing` and `packing` as needed. Table 3 uses
the same interface through `all_models.py`, with `bf16 own transfer dfloat11`.
Use `--campaign 2` and a new directory for an independent repeat; fresh commands
measure the current implementation, not the historical baseline revision.

Protocol: batch one, eager attention, no cache/offload, 20 warmups, five
synchronized 50-forward trials. Export/initialization/compilation/correctness
are outside timing. Decoder-only timing is separate. Selected weights and
outputs must be bitwise exact; setup/input must stay frozen. Live allocation,
forward peak and reservation are separate. No decoded full-model cache is kept.
The benchmark refuses to use a GPU occupied by another process.

`python scripts/six_models_ablations.py --check-kernels --gpu 0` runs the
synthetic CUDA boundaries/tails/long-code checks and supported-module/save-reload
checks. It is a correctness run, not an inference timing campaign.

## Preliminaries and optimizers

```bash
python scripts/preliminaries.py --models dinov3-vit --verify-published \
  --output results/reruns/prelim-vit
CUDA_VISIBLE_DEVICES=0 python scripts/optimizers.py --models mlp \
  --optimizers Adam Muon --seeds 17 --steps 3 --verify-published \
  --output results/reruns/optimizer-prefix
```

Preliminaries use full BF16 field histograms, the recorded 8 Mi-element sampling
scheme for n-grams, and the original seeded 48-chunk JS selection. Integer
populations and sampled input hashes must match exactly; entropy/JS reductions
allow `1e-12` floating-point tolerance.

The optimizer study uses an MLP, CIFAR-stem ResNet18, real ViT-B/16 and real
GPT-2—not the earlier toy models. SGD, Adam, AdamW and native PyTorch Muon use
the recorded hyperparameters, seeds 17/29/43, matched initialization/minibatches,
and 1,000 updates. FP32 training is unchanged by measuring a new BF16 copy after
each step. Raw 256-bin histograms, losses, initial/final state hashes and data
order are saved; CSV exports include the three-seed mean and standard deviation.
`--verify-published` can check a short prefix without training all 48 cases.

The restructuring was spot-checked on one DINOv3 ViT CPU/GPU control,
ResNet101 own/transfer/ZipNN, Pythia's final transfer/ZipNN point, DINOv3 ViT
preliminaries and MLP Adam/Muon's first three steps. Compression archives and
optimizer histograms matched the saved evidence; fresh timings are explicitly
new measurements. See `paper/verification.json` for the checks and their scope.
