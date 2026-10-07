# Paper evidence

`paper/godel-original-metadata.zip` contains 553 original CPU campaign metadata
and source files recovered directly from Godel. The companion
`paper/godel-original-metadata-members.json` records every member's SHA256 and
size, verified against the remote original, plus the archive SHA256.

`paper/gx01-original-metadata.zip` through `paper/gx04-original-metadata.zip`
contain 2,269 GPU campaign metadata files recovered from the four GX hosts.
Each has a companion `*-members.json` with remote-verified member hashes,
archive SHA256 and all 30 captured runtime hashes reverified live on that host.
The records include completed results, failed attempts and diagnostic history;
not every archived record is an accepted paper measurement. No GPU experiment
was rerun and numerical cohort validation was not repeated during recovery.

`paper/catalogue.json` contains pinned input identities and tensor manifests.
Fresh outputs go under `scripts/results/reruns/` (ignored by Git).

Verify the recovered archives without loading tensors or running experiments:

```bash
python - <<'PY'
import hashlib, json, pathlib, zipfile
root = pathlib.Path('scripts/results/paper')
for manifest in sorted(root.glob('*-original-metadata-members.json')):
    report = json.loads(manifest.read_text())
    archive = manifest.with_name(manifest.name.replace('-members.json', '.zip'))
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == report['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist()) == set(report['members'])
        for name, entry in report['members'].items():
            data = z.read(name)
            assert len(data) == entry['bytes']
            assert hashlib.sha256(data).hexdigest() == entry['sha256']
    print('PASS', archive.name, len(report['members']))
PY
```
