"""Stable host and CPU metadata shared by experiment runners."""

import hashlib
import os
from pathlib import Path
import platform


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
            continue
        if "-" in part:
            first, last = part.split("-", 1)
            cpus.update(range(int(first), int(last) + 1))
        else:
            cpus.add(int(part))
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
            hashlib.sha256(machine_id.read_bytes()).hexdigest()
            if machine_id.is_file()
            else None
        )
    return facts
