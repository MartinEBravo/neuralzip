# GB10 four-machine campaign: verified snapshot

This snapshot contains one complete ResNet101 GPU cohort. It supplements, and
does not replace, `../paper/evidence.zip`. The other campaign models were still
running when this snapshot was saved. No CPU results changed.

| Method | Inference (ms) | Live allocation (bytes) | Forward peak (bytes) |
|---|---:|---:|---:|
| BF16 | 14.733344297856092 | 99906048 | 161436160 |
| NeuralZip own | 16.474788328632712 | 76719616 | 138249728 |
| NeuralZip transfer | 16.371346367523074 | 78224896 | 139755008 |
| DFloat11 | 15.784406401216984 | 75075584 | 137244672 |

All four methods used the same NVIDIA GB10, pinned tensors, selected weights,
forward inputs and software sources. Batch one, 20 warmups, five synchronized
50-forward trials; the reported latency is the median. Every timed trial had
unchanged system swap-in/out counters. Weights/outputs were bitwise exact;
inputs and setups stayed unchanged. Initialization and compression are outside
GPU inference timing. No retained decoded full-model weights or offload.

`evidence.zip` contains the four original numerical records, four timing guard
records, completion manifest and independent verification report. Only user
paths were anonymized; the completion manifest references the distributed
records. `evidence-members.json` retains their original and distributed SHA256
values. No tensor, setup pickle, dependency or credential files are included.

Verify the distributed archive from the repository root:

```bash
python - <<'PY'
import hashlib, json, pathlib, zipfile
p = pathlib.Path('results/gb10-four-20261006')
members = json.loads((p / 'evidence-members.json').read_text())
with zipfile.ZipFile(p / 'evidence.zip') as z:
    assert z.testzip() is None
    assert set(z.namelist()) == set(members)
    for name, entry in members.items():
        assert hashlib.sha256(z.read(name)).hexdigest() == entry['sha256']
    assert json.loads(z.read('verification.json'))['independent_checks_passed']
print('PASS: complete ResNet101 GB10 evidence')
PY
```

This snapshot is not confirmation that the all-GB10 paper update is complete.
