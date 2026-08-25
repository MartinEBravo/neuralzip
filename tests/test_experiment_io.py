import csv
import json

from neuralzip.experiment_io import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    read_csv,
)


def test_atomic_writers_replace_temporary_files_and_preserve_dialects(tmp_path):
    text_path = tmp_path / "nested" / "value.txt"
    json_path = tmp_path / "nested" / "value.json"
    crlf_path = tmp_path / "nested" / "crlf.csv"
    lf_path = tmp_path / "nested" / "lf.csv"

    atomic_write_text(text_path, "first")
    atomic_write_text(text_path, "second")
    atomic_write_json(json_path, {"value": 2})
    compact_path = tmp_path / "nested" / "compact.json"
    atomic_write_json(compact_path, {"value": 2}, trailing_newline=False)
    atomic_write_csv(crlf_path, ("name",), ({"name": "value"},))
    atomic_write_csv(
        lf_path,
        ("name",),
        ({"name": "value"},),
        lineterminator="\n",
    )

    assert text_path.read_text() == "second"
    assert json.loads(json_path.read_text()) == {"value": 2}
    assert not compact_path.read_bytes().endswith(b"\n")
    assert crlf_path.read_bytes() == b"name\r\nvalue\r\n"
    assert lf_path.read_bytes() == b"name\nvalue\n"
    assert read_csv(lf_path) == [{"name": "value"}]
    assert not list((tmp_path / "nested").glob("*.tmp"))


def test_csv_projection_preserves_field_order_and_missing_values(tmp_path):
    path = tmp_path / "rows.csv"
    atomic_write_csv(
        path,
        ("second", "first"),
        ({"first": "a", "ignored": "x"},),
        lineterminator="\n",
    )

    with path.open(newline="") as handle:
        assert list(csv.reader(handle)) == [["second", "first"], ["", "a"]]
