"""Stable host and CPU metadata shared by experiment runners."""

import importlib
import importlib.metadata
import os
import platform
from pathlib import Path

from .hashing import sha256_file


def package_version(name):
    """Return an installed distribution version, or ``None`` when absent."""

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def package_versions(*names):
    """Return installed versions for each distribution name."""
    return {name: package_version(name) for name in names}


def module_path(name):
    """Return the resolved file backing an imported module."""
    return Path(str(importlib.import_module(name).__file__)).resolve()


def cpu_affinity():
    """Return the process CPU affinity as a sorted list when supported."""

    if not hasattr(os, "sched_getaffinity"):
        return None
    return sorted(int(cpu) for cpu in os.sched_getaffinity(0))


def parse_cpu_set(value):
    """Parse Linux CPU-list syntax such as ``0-3,8,10-11``."""

    cpus = set()
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            raise ValueError("CPU set contains an empty segment.")
        if "-" in part:
            first, last = part.split("-", 1)
            first_cpu = int(first)
            last_cpu = int(last)
            if first_cpu < 0 or last_cpu < first_cpu:
                raise ValueError(f"Invalid CPU range: {part!r}.")
            cpus.update(range(first_cpu, last_cpu + 1))
        else:
            cpu = int(part)
            if cpu < 0:
                raise ValueError(f"CPU index cannot be negative: {cpu}.")
            cpus.add(cpu)
    if not cpus:
        raise ValueError("CPU set cannot be empty.")
    return cpus


def cpu_model():
    """Return the platform CPU description, including Linux model names."""

    description = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return description


def host_facts(*, include_machine_id=False):
    """Return deterministic host fields used in experiment provenance."""

    facts = {
        "hostname": platform.node(),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "cpu_model": cpu_model(),
        "logical_cpus": os.cpu_count(),
    }
    if include_machine_id:
        machine_id = Path("/etc/machine-id")
        facts["machine_id_sha256"] = (
            sha256_file(machine_id) if machine_id.is_file() else None
        )
    return facts
