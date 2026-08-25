"""Small atomic I/O primitives shared by experiment runners."""

import csv
import json
from pathlib import Path


def atomic_write_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(
    path,
    value,
    *,
    sort_keys=False,
    trailing_newline=True,
):
    serialized = json.dumps(
        value,
        indent=2,
        sort_keys=sort_keys,
        default=str,
    )
    if trailing_newline:
        serialized += "\n"
    atomic_write_text(path, serialized)


def read_csv(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def atomic_write_csv(
    path,
    fields,
    rows,
    *,
    lineterminator=None,
):
    path = Path(path)
    fields = tuple(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    writer_options = {}
    if lineterminator is not None:
        writer_options["lineterminator"] = lineterminator
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, **writer_options)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    temporary.replace(path)
