# Paper evidence

`paper/godel-original-metadata.zip` contains 553 original CPU campaign metadata
and source files recovered directly from Godel. The companion
`paper/godel-original-metadata-members.json` records every member's SHA256 and
size, verified against the remote original, plus the archive SHA256.
The archive contains JSON and Python only: no weights or setup pickles.
Numerical cohort validation has not been rerun during recovery.

Final GX GPU evidence is still missing: all four hosts timed out during recovery.
This recovery archive does not use the legacy `evidence.zip` schema;
`--check` and `--export-saved` remain unavailable pending final integration.

`paper/catalogue.json` contains pinned input identities and tensor manifests.
Fresh outputs go under `scripts/results/reruns/` (ignored by Git).
