"""The measured source-only bank and gate. The native codec is unchanged."""

from dataclasses import replace

import numpy as np
import torch

from neuralzip import huff0
from neuralzip.compressor import NeuralZipCompressor
from neuralzip.payloads import CodebookPayload
from neuralzip.setup import NeuralZipSetup
from neuralzip.utils import time_call


def exponents(tensor):
    bits = tensor.reshape(-1).view(torch.uint16).numpy()
    return ((bits >> 7) & 255).astype(np.uint8)


def sampled_histograms(tensor, step, budget=1024):
    """A fixed-stride routing sample, never a codebook training histogram."""
    bits = tensor.reshape(-1).view(torch.uint16).numpy()
    full, tail = divmod(len(bits), step)
    stride = max(1, step // budget)
    rows = []
    if full:
        samples = (
            (bits[: full * step].reshape(full, step)[:, ::stride] >> 7) & 255
        ).astype(np.int64)
        samples += np.arange(full, dtype=np.int64)[:, None] * 256
        rows.append(
            np.bincount(samples.ravel(), minlength=full * 256).reshape(full, 256)
        )
    if tail:
        samples = (bits[full * step :: max(1, tail // budget)] >> 7) & 255
        rows.append(np.bincount(samples, minlength=256)[None, :])
    return np.concatenate(rows).astype(np.float32)


def greedy_bits(values, lengths, packed_lengths):
    """Exact frozen-table cost of a short sample; never constructs a table."""
    total, position = 0, 0
    while position < len(values):
        first = int(values[position])
        packed = (
            int(packed_lengths[first | (int(values[position + 1]) << 8)])
            if packed_lengths is not None and position + 1 < len(values)
            else 0
        )
        total += packed if packed else int(lengths[first])
        position += 2 if packed else 1
    return total


class BankCompressor(NeuralZipCompressor):
    routing_audit = []
    gate_cost_audit = []

    def __init__(self, setup=None, **kwargs):
        super().__init__(setup, **kwargs)
        self.bank_ids = getattr(setup, "bank_ids", ())
        self.bank_lengths = np.asarray(
            getattr(setup, "bank_lengths", ()), dtype=np.float32
        )
        by_id = (
            {book.codebook_id: book for book in setup.codebooks}
            if self.bank_ids
            else {}
        )
        self.bank_modes = tuple(
            dict(
                kind=by_id[i].mode,
                codebook_id=i,
                codebook=huff0.build_codebook_from_header(
                    by_id[i].codebook, vocab=by_id[i].vocab
                ),
            )
            for i in self.bank_ids
        )
        self.fallback_ids = getattr(setup, "bank_fallback_ids", ())
        self.routing_candidates = getattr(setup, "bank_routing_candidates", 1)
        self.trigger_bits = getattr(setup, "bank_trigger_bits", None)
        self.gate_algorithm = getattr(setup, "bank_gate_algorithm", "")
        self.greedy_gate = self.gate_algorithm in ("weighted_greedy", "volume_greedy")
        self.gate_costs = {}
        if self.trigger_bits is not None:
            symbols = np.arange(65536)
            for i in setup.bank_default_ids:
                book = by_id[i]
                bits = (
                    np.frombuffer(book.codebook[4:], dtype=np.uint8)
                    if book.vocab
                    else np.frombuffer(
                        huff0.export_code_table(book.codebook)[0], dtype=np.uint8
                    )
                )
                if self.greedy_gate:
                    packed = None
                    if book.vocab:
                        packed = np.zeros(65536, dtype=np.uint8)
                        packed[np.frombuffer(book.vocab, dtype="<u2")] = bits[256:]
                    cost = bits[:256], packed
                else:
                    cost = bits[symbols & 255].astype(np.uint16) + bits[symbols >> 8]
                    if book.vocab:
                        cost[np.frombuffer(book.vocab, dtype="<u2")] = bits[256:]
                self.gate_costs[i] = cost
        self.fallback_modes = tuple(
            dict(
                kind="shared_huff0",
                codebook_id=i,
                codebook=huff0.build_codebook_from_header(by_id[i].codebook),
            )
            for i in self.fallback_ids
        )
        if self.fallback_ids:
            symbols = np.arange(65536)
            self.pair_costs = []
            for i in self.bank_ids:
                book = by_id[i]
                if book.vocab:
                    bits = np.frombuffer(book.codebook[4:], dtype=np.uint8)
                    cost = bits[symbols & 255].astype(np.uint16) + bits[symbols >> 8]
                    cost[np.frombuffer(book.vocab, dtype="<u2")] = bits[256:]
                else:
                    bits = np.frombuffer(
                        huff0.export_code_table(book.codebook)[0], dtype=np.uint8
                    )
                    cost = bits[symbols & 255].astype(np.uint16) + bits[symbols >> 8]
                self.pair_costs.append(cost)
            self.pair_costs = np.asarray(self.pair_costs, dtype=np.uint16)

    def prepare_storage_transfer_target(self, named_tensors):
        super().prepare_storage_transfer_target(named_tensors)
        if not self.bank_ids:
            return
        if self.trigger_bits is not None:
            trigger = False
            weighted_cost, weight = 0.0, 0
            if self.gate_algorithm == "volume_greedy":
                sizes = np.asarray(
                    [t.numel() for _, t in named_tensors], dtype=np.int64
                )
                cumulative = np.cumsum(sizes)
                points = (np.arange(16) + 0.5) * cumulative[-1] / 16
                probes = []
                for point, index in zip(points, np.searchsorted(cumulative, points)):
                    name, tensor = named_tensors[int(index)]
                    local = point - (cumulative[index - 1] if index else 0)
                    probes.append((name, tensor, int(local)))
            else:
                probes = [
                    (name, tensor, None)
                    for name, tensor in sorted(
                        named_tensors, key=lambda item: item[1].numel(), reverse=True
                    )[:8]
                ]
            for name, tensor, local in probes:
                mode = self._transfer_mode_by_tensor_stream[
                    self._transfer_plan_key(name, tensor, 1)
                ]
                if mode["kind"] == "raw":
                    if not self.greedy_gate:
                        trigger = True
                        break
                    factor = 1 if local is not None else tensor.numel()
                    weighted_cost += 8 * factor
                    weight += factor
                    continue
                raw = tensor.reshape(-1).view(torch.uint16).numpy()
                if self.greedy_gate:
                    window = min(256 if local is not None else 64, len(raw))
                    starts = (
                        [max(0, min(len(raw) - window, local - window // 2))]
                        if local is not None
                        else np.linspace(0, len(raw) - window, 8, dtype=int)
                    )
                    lengths, packed = self.gate_costs[mode["codebook_id"]]
                    bits = sum(
                        greedy_bits(
                            ((raw[s : s + window] >> 7) & 255).astype(np.uint8),
                            lengths,
                            packed,
                        )
                        for s in starts
                    )
                    factor = 1 if local is not None else tensor.numel()
                    weighted_cost += min(8, bits / (len(starts) * window)) * factor
                    weight += factor
                    continue
                starts = np.linspace(
                    0, len(raw) - 2, min(1024, len(raw) // 2), dtype=int
                )
                first, second = (raw[starts] >> 7) & 255, (raw[starts + 1] >> 7) & 255
                pairs = first.astype(np.int64) | (second.astype(np.int64) << 8)
                if (
                    len(pairs)
                    and self.gate_costs[mode["codebook_id"]][pairs].mean() / 2
                    > self.trigger_bits
                ):
                    trigger = True
                    break
            if self.greedy_gate:
                predicted = weighted_cost / weight
                trigger = predicted > self.trigger_bits
                type(self).gate_cost_audit.append(predicted)
            self.robust_routing_used = trigger
            type(self).routing_audit.append(trigger)
            if not trigger:
                return
        step = self.chunk_size // 2
        for name, tensor in named_tensors:
            assert tensor.dtype == torch.bfloat16
            histogram = sampled_histograms(tensor, step)
            scores = histogram @ self.bank_lengths.T
            selected = np.argmin(scores, axis=1)
            key = self._transfer_plan_key(name, tensor, 1)
            modes = [self.bank_modes[int(i)] for i in selected]
            if self.fallback_ids:
                # Compare already-frozen tables on adjacent routing samples; no code learning.
                raw = tensor.reshape(-1).view(torch.uint16).numpy()
                k = min(self.routing_candidates, len(self.bank_ids))
                candidates = np.argpartition(scores, k - 1, axis=1)[:, :k]
                full, tail = divmod(len(raw), step)
                pair_samples = []
                if full:
                    parts = raw[: full * step].reshape(full, step)
                    stride = max(2, step // 1024)
                    first, second = (
                        (parts[:, 0 : step - 1 : stride] >> 7) & 255,
                        (parts[:, 1:step:stride] >> 7) & 255,
                    )
                    pair_samples.append(
                        first.astype(np.int64) | (second.astype(np.int64) << 8)
                    )
                if tail > 1:
                    part = raw[full * step :]
                    starts = np.arange(0, tail - 1, max(2, tail // 1024))
                    first, second = (
                        (part[starts] >> 7) & 255,
                        (part[starts + 1] >> 7) & 255,
                    )
                    pair_samples.append(
                        (first.astype(np.int64) | (second.astype(np.int64) << 8))[
                            None, :
                        ]
                    )
                offset = 0
                for samples in pair_samples:
                    count = len(samples)
                    shortlist = candidates[offset : offset + count]
                    estimates = np.stack(
                        [
                            self.pair_costs.ravel()[
                                shortlist[:, rank, None] * 65536 + samples
                            ].mean(axis=1)
                            / 2
                            for rank in range(k)
                        ],
                        axis=1,
                    )
                    best_rank = np.argmin(estimates, axis=1)
                    best = shortlist[np.arange(count), best_rank]
                    single_choice = selected[offset : offset + count]
                    single_bits = scores[offset : offset + count][
                        np.arange(count), single_choice
                    ] / histogram[offset : offset + count].sum(axis=1)
                    use_mixed = estimates[np.arange(count), best_rank] <= single_bits
                    modes[offset : offset + count] = [
                        self.bank_modes[int(m)]
                        if packed
                        else self.fallback_modes[int(s)]
                        for m, s, packed in zip(best, single_choice, use_mixed)
                    ]
                    offset += count
            self._transfer_chunk_modes[key] = tuple(modes)


def build_bank(named, gathered, gate_bits=3.0):
    variant = "bank-physical-v5"
    setup = NeuralZipSetup()
    fractions = (
        tuple(i / 8 for i in range(8))
        if variant in ("bank-physical-v4", "bank-physical-v5")
        else ((0.0, 0.25, 0.5, 0.75) if variant != "bank" else (0.0,))
    )
    physical = variant.startswith("bank-physical")
    packed = variant.startswith("bank-packed") or physical

    def build():
        (groups, assignments, parents), cluster_seconds = time_call(
            setup._cluster_chunks, named
        )
        (books, chunks, tensors, report), book_seconds = time_call(
            setup._build_codebooks, groups, assignments, parents
        )
        setup.codebooks, setup.chunk_codebooks, setup.tensor_codebooks = (
            tuple(books),
            {},
            tensors,
        )
        setup.packing_report = ()
        ids, lengths, fallbacks, cache, single_cache = [], [], [], {}, {}
        eligible = sorted(
            range(len(groups)), key=lambda i: groups[i]["weight"], reverse=True
        )[:8]
        calibrated = {}
        if physical:
            for fraction in fractions:
                collected = [[] for _ in groups]
                sampled = [0] * len(groups)
                threshold = int(np.ceil(128 * (2 ** (1 - fraction) - 1)))
                for name, tensor in named:
                    raw = tensor.reshape(-1).view(torch.uint16).numpy()
                    group_ids = assignments[(name, "bfloat16", 1, tensor.numel())]
                    for chunk, group_id in enumerate(group_ids):
                        if group_id not in eligible:
                            continue
                        budget = setup.max_codebook_sample_bytes - sampled[group_id]
                        if budget <= 0:
                            continue
                        part = raw[
                            chunk * (setup.chunk_size // 2) : (chunk + 1)
                            * (setup.chunk_size // 2)
                        ][:budget]
                        values = ((part >> 7) & 255).astype(np.uint8)
                        bump = (
                            ((part & 127) >= threshold) & (values > 0) & (values < 254)
                        )
                        values[bump] += np.uint8(1)
                        collected[group_id].append(values.tobytes())
                        sampled[group_id] += len(values)
                calibrated[fraction] = collected
        for group_id in eligible:
            group = groups[group_id]
            counts = np.asarray(
                huff0.count_mixed_tokens(group["samples"], b""), dtype=np.int64
            )
            for fraction in fractions:
                augmented = calibrated[fraction][group_id] if physical else None
                blended = (
                    np.asarray(huff0.count_mixed_tokens(augmented, b""), dtype=np.int64)
                    if physical
                    else np.rint(
                        (1 - fraction) * counts + fraction * np.roll(counts, 1)
                    ).astype(np.int64)
                )
                packed_book, vocab = None, b""
                if packed and group["sampled"] >= 4096:
                    if not physical:
                        augmented = []
                        for sample in group["samples"]:
                            values = np.frombuffer(sample, dtype=np.uint8).copy()
                            shift_mask = (np.arange(len(values)) % 4) < int(
                                4 * fraction
                            )
                            values[shift_mask] += np.uint8(1)
                            augmented.append(values.tobytes())
                    frequencies = np.zeros(65536, dtype=np.int64)
                    for sample in augmented:
                        values = np.frombuffer(sample, dtype=np.uint8).astype(np.uint16)
                        frequencies += np.bincount(
                            values[:-1] | (values[1:] << 8), minlength=65536
                        )
                    observed = np.flatnonzero(frequencies)
                    selected = np.sort(
                        observed[
                            np.argsort(-frequencies[observed], kind="stable")[:254]
                        ]
                    ).astype("<u2")
                    vocab = selected.tobytes()
                    token_counts = np.asarray(
                        huff0.count_mixed_tokens(augmented, vocab), dtype=np.int64
                    )
                    token_counts[:256] += np.rint(0.01 * blended).astype(np.int64)
                    packed_book = huff0.build_mixed_codebook(
                        token_counts, vocab, prior_count=1
                    )
                for shift in range(-8, 9):
                    shifted = np.roll(blended, shift)
                    try:
                        single = huff0.build_codebook_from_counts(
                            shifted, prior_count=1, max_bits=12
                        )
                    except RuntimeError:
                        continue
                    table = np.frombuffer(
                        huff0.export_code_table(single.codebook)[0], dtype=np.uint8
                    ).copy()
                    assert len(table) == 256 and np.all(table > 0)
                    header, shifted_vocab, mode = single.codebook, b"", "shared_huff0"
                    if packed_book is not None:
                        codes = np.frombuffer(packed_book.codebook[4:], dtype=np.uint8)
                        packed_values = np.frombuffer(vocab, dtype="<u2")
                        shifted_values = (
                            ((packed_values & 255).astype(np.int32) + shift) % 256
                            | (
                                (((packed_values >> 8).astype(np.int32) + shift) % 256)
                                << 8
                            )
                        ).astype("<u2")
                        order = np.argsort(shifted_values)
                        shifted_vocab = shifted_values[order].tobytes()
                        header = (
                            b"NZM1"
                            + np.concatenate(
                                (np.roll(codes[:256], shift), codes[256:][order])
                            ).tobytes()
                        )
                        # Activation validates the immutable transformed header, not target weights.
                        huff0.build_codebook_from_header(header, vocab=shifted_vocab)
                        mode = "mixed_huffman"
                    identity = table.tobytes() if physical else (header, shifted_vocab)
                    if identity in cache:
                        continue
                    book_id = len(setup.codebooks)
                    setup.codebooks += (
                        CodebookPayload(
                            book_id, 1, "bfloat16", mode, header, shifted_vocab
                        ),
                    )
                    ids.append(book_id)
                    lengths.append(table.tolist())
                    cache[identity] = (len(groups), shift, fraction)
                    if physical:
                        if single.codebook not in single_cache:
                            fallback_id = len(setup.codebooks)
                            setup.codebooks += (
                                CodebookPayload(
                                    fallback_id, 1, "bfloat16", codebook=single.codebook
                                ),
                            )
                            single_cache[single.codebook] = fallback_id
                        fallbacks.append(single_cache[single.codebook])
        setup.bank_ids, setup.bank_lengths = tuple(ids), tuple(lengths)
        setup.bank_fallback_ids = tuple(fallbacks)
        setup.bank_routing_candidates = (
            4 if variant in ("bank-physical-v4", "bank-physical-v5") else 1
        )
        setup.source_cluster_count = len(groups)
        assert ids
        if variant == "bank-physical-v5":
            frozen = gathered
            source_books = list(setup.codebooks)
            used = set(setup.bank_ids) | set(setup.bank_fallback_ids)
            extra = [book for book in source_books if book.codebook_id in used]
            offset = max(book.codebook_id for book in frozen.codebooks) + 1
            remap = {book.codebook_id: offset + i for i, book in enumerate(extra)}
            setup.codebooks = frozen.codebooks + tuple(
                replace(book, codebook_id=remap[book.codebook_id]) for book in extra
            )
            setup.tensor_codebooks = frozen.tensor_codebooks
            setup.global_codebooks = frozen.global_codebooks
            setup.bank_ids = tuple(remap[i] for i in setup.bank_ids)
            setup.bank_fallback_ids = tuple(remap[i] for i in setup.bank_fallback_ids)
            setup.bank_default_ids = tuple(
                book.codebook_id for book in frozen.codebooks
            )
            setup.bank_trigger_bits = 4.0
        return cluster_seconds, book_seconds

    (clustering, core_books), seconds = time_call(build)
    setup.bank_gate_algorithm = "volume_greedy"
    setup.bank_trigger_bits = float(gate_bits)
    assert not setup.chunk_codebooks
    return setup, seconds
