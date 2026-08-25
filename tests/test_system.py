import pytest

from neuralzip import system


def test_parse_cpu_set_supports_ranges_and_rejects_invalid_values():
    assert system.parse_cpu_set("0-2, 5,7-8") == {0, 1, 2, 5, 7, 8}
    for value in ("", " , ", "0,,2", "-1", "3-1"):
        with pytest.raises(ValueError):
            system.parse_cpu_set(value)


def test_cpu_affinity_is_sorted_and_platform_optional(monkeypatch):
    monkeypatch.setattr(system.os, "sched_getaffinity", lambda _pid: {4, 1, 3}, raising=False)
    assert system.cpu_affinity() == [1, 3, 4]
    monkeypatch.delattr(system.os, "sched_getaffinity")
    assert system.cpu_affinity() is None


def test_package_version_returns_none_when_distribution_is_missing(monkeypatch):
    def missing(_name):
        raise system.importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(system.importlib.metadata, "version", missing)
    assert system.package_version("not-installed") is None


def test_host_facts_adds_machine_digest_only_when_requested(tmp_path, monkeypatch):
    monkeypatch.setattr(system, "cpu_model", lambda: "test-cpu")
    facts = system.host_facts()
    assert facts["cpu_model"] == "test-cpu"
    assert "machine_id_sha256" not in facts

    with_digest = system.host_facts(include_machine_id=True)
    assert "machine_id_sha256" in with_digest
