// Copyright 2025 Tianyi Zhang
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Modified for NeuralZip in 2026: specialized singleton/pair LUTs, bounded
// lookahead, chunk descriptors, and a two-barrier hierarchical warp scan.

// NVRTC may not expose host C headers; CUDA primitive widths are fixed.
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef int int32_t;

#define THREAD_ID           threadIdx.x
#define BLOCK_ID            blockIdx.x
#define N_THREADS           blockDim.x
#define BYTES_PER_THREAD    8

namespace {

__device__ __forceinline__ uint8_t read_packed_gap(
    const uint8_t* __restrict__ gaps,
    const uint64_t thread_id) {
    // Four-bit gaps are packed most-significant nibble first.
    const uint8_t packed = gaps[thread_id >> 1U];
    const unsigned shift = (thread_id & 1U) == 0U ? 4U : 0U;
    return static_cast<uint8_t>((packed >> shift) & 0x0FU);
}

__device__ __forceinline__ uint32_t decode_lut_entry(
    const uint32_t* __restrict__ lut,
    const uint64_t bit_buffer,
    const int max_bits) {
    const uint32_t index = static_cast<uint32_t>(
        bit_buffer >> (64U - static_cast<unsigned>(max_bits)));
    uint32_t entry = __ldg(lut + index);
    for (unsigned depth = max_bits; (entry >> 24U) == 0U; ++depth) {
        const unsigned branch = (bit_buffer >> (63U - depth)) & 1U;
        entry = __ldg(lut + entry + branch);
    }
    return entry;
}

__device__ __forceinline__ uint16_t decode_lut_entry(
    const uint16_t* __restrict__ lut,
    const uint64_t bit_buffer,
    const int max_bits) {
    return __ldg(lut + static_cast<uint32_t>(bit_buffer >> (64U - max_bits)));
}

__device__ __forceinline__ uint8_t entry_nbits(const uint32_t entry) {
    return static_cast<uint8_t>(entry >> 24U);
}

__device__ __forceinline__ uint8_t entry_nbits(const uint16_t entry) {
    return static_cast<uint8_t>(entry >> 8U);
}

__device__ __forceinline__ uint32_t entry_count(const uint32_t entry) {
    return (entry >> 16U) & 255U;
}

__device__ __forceinline__ uint32_t entry_count(const uint16_t entry) {
    return 1U;
}

__device__ __forceinline__ uint64_t peek_bits(
    uint64_t high, uint64_t low, unsigned consumed) {
    return consumed ? (high << consumed) | (low >> (64U - consumed)) : high;
}

template <typename Word, bool Reorder, int MantissaBits, int ExponentBits, int StreamId>
__device__ __forceinline__ Word assemble_value(
    const uint8_t symbol,
    const uint8_t* const* raw_streams,
    const uint32_t index) {
    uint32_t value = 0;
#pragma unroll
    for (int lane = 0; lane < sizeof(Word); ++lane) {
        const uint8_t byte = lane == StreamId ? symbol : raw_streams[lane][index];
        value |= static_cast<uint32_t>(byte) << (8U * lane);
    }
    if (Reorder) {
        value = (value & ((1U << MantissaBits) - 1U))
            | ((value << ExponentBits) & (1U << (MantissaBits + ExponentBits)))
            | ((value >> 1U) & (((1U << ExponentBits) - 1U) << MantissaBits));
    }
    return static_cast<Word>(value);
}

template <typename Word, bool Reorder, int MantissaBits, int ExponentBits, int StreamId, bool StageSymbols, typename Entry>
__device__ __forceinline__ uint32_t write_entry(
    const Entry entry,
    Word* __restrict__ buffer,
    const uint8_t* const* raw_streams,
    uint32_t index,
    const uint32_t offset,
    const uint32_t end) {
    if (StageSymbols) {
        reinterpret_cast<uint8_t*>(buffer)[index - offset] = entry & 255U;
    } else {
        buffer[index - offset] = assemble_value<Word, Reorder, MantissaBits, ExponentBits, StreamId>(
            entry & 255U, raw_streams, index);
    }
    ++index;
    if (sizeof(Entry) == 4 && entry_count(entry) == 2U && index < end) {
        if (StageSymbols) {
            reinterpret_cast<uint8_t*>(buffer)[index - offset] = (entry >> 8U) & 255U;
        } else {
            buffer[index - offset] = assemble_value<Word, Reorder, MantissaBits, ExponentBits, StreamId>(
                (entry >> 8U) & 255U, raw_streams, index);
        }
        ++index;
    }
    return index;
}

__device__ __forceinline__ uint32_t exclusive_block_prefix(
    const uint32_t value,
    uint32_t* __restrict__ warp_offsets) {
    // Scan inside each warp, then let warp zero scan the per-warp totals.
    const uint32_t lane = THREAD_ID & 31U;
    const uint32_t warp_id = THREAD_ID >> 5U;
    const uint32_t warp_count = (N_THREADS + 31U) >> 5U;
    const uint32_t active_mask = __activemask();

    uint32_t inclusive = value;
#pragma unroll
    for (uint32_t offset = 1U; offset < 32U; offset <<= 1U) {
        const uint32_t preceding = __shfl_up_sync(
            active_mask,
            inclusive,
            offset);
        if (lane >= offset) {
            inclusive += preceding;
        }
    }

    if (lane + 1U == static_cast<uint32_t>(__popc(active_mask))) {
        warp_offsets[warp_id] = inclusive;
    }
    __syncthreads();

    if (warp_id == 0U) {
        uint32_t warp_total = lane < warp_count ? warp_offsets[lane] : 0U;
        const uint32_t own_total = warp_total;
#pragma unroll
        for (uint32_t offset = 1U; offset < 32U; offset <<= 1U) {
            const uint32_t preceding = __shfl_up_sync(
                active_mask,
                warp_total,
                offset);
            if (lane >= offset) {
                warp_total += preceding;
            }
        }
        if (lane < warp_count) {
            warp_offsets[lane] = warp_total - own_total;
        }
    }
    __syncthreads();

    return warp_offsets[warp_id] + inclusive - value;
}

template <typename Word, bool Reorder, int MantissaBits, int ExponentBits, int StreamId, bool Mixed, bool StageSymbols, typename Entry>
__device__ __forceinline__ void decode_short_block(
    const Entry* __restrict__ lut,
    const int max_bits,
    const uint8_t* __restrict__ codes,
    const uint8_t* const* raw_streams,
    const uint8_t* __restrict__ gaps,
    const uint32_t* __restrict__ output_positions,
    Word* __restrict__ outputs,
    const int n_bytes,
    const uint32_t local_block_id,
    uint8_t* __restrict__ shared_mem) {
    uint32_t* warp_offsets = reinterpret_cast<uint32_t*>(shared_mem);
    const uint32_t warp_count = (N_THREADS + 31U) >> 5U;
    const uint64_t group_thread_id =
        static_cast<uint64_t>(local_block_id) * N_THREADS + THREAD_ID;
    const uint64_t code_offset = group_thread_id * BYTES_PER_THREAD;

    // Codes of at most 32 bits need only one refill per 64-bit segment.
    uint64_t high = 0;
    uint32_t lookahead = 0;
#pragma unroll
    for (int index = 0; index < 12; ++index) {
        const uint64_t source = code_offset + index;
        const uint64_t byte = source < static_cast<uint64_t>(n_bytes) ? codes[source] : 0U;
        if (index < 8) high |= byte << (56U - 8U * index);
        else lookahead |= static_cast<uint32_t>(byte) << (24U - 8U * (index - 8));
    }
    const uint8_t gap = Mixed ? gaps[group_thread_id] : read_packed_gap(gaps, group_thread_id);
    uint64_t buffer = high << gap;
    unsigned free_bits = code_offset < static_cast<uint64_t>(n_bytes) ? gap : 64U;
    uint32_t thread_counter = 0;
    while (free_bits < 32U) {
        const Entry entry = decode_lut_entry(lut, buffer, max_bits);
        thread_counter += entry_count(entry);
        buffer <<= entry_nbits(entry);
        free_bits += entry_nbits(entry);
    }
    buffer |= static_cast<uint64_t>(lookahead) << (free_bits - 32U);
    free_bits -= 32U;
    while (free_bits < 32U) {
        const Entry entry = decode_lut_entry(lut, buffer, max_bits);
        thread_counter += entry_count(entry);
        buffer <<= entry_nbits(entry);
        free_bits += entry_nbits(entry);
    }

    const uint32_t write_offset = __ldg(output_positions + local_block_id);
    const uint32_t block_end = __ldg(output_positions + local_block_id + 1U);
    uint32_t output_idx = write_offset
        + exclusive_block_prefix(thread_counter, warp_offsets);
    uint64_t requested_end =
        static_cast<uint64_t>(output_idx) + thread_counter;
    if (requested_end > block_end) {
        requested_end = block_end;
    }
    const uint32_t end_output_idx = static_cast<uint32_t>(requested_end);

    // Stage only exponent bytes, not dense words. Large groups retain direct writes.
    Word* write_buffer = StageSymbols ? reinterpret_cast<Word*>(
        shared_mem + static_cast<uint64_t>(warp_count) * sizeof(uint32_t)) : outputs + write_offset;
    buffer = high << gap;
    free_bits = gap;
    while (free_bits < 32U && output_idx < end_output_idx) {
        const Entry entry = decode_lut_entry(lut, buffer, max_bits);
        output_idx = write_entry<Word, Reorder, MantissaBits, ExponentBits, StreamId, StageSymbols>(
            entry, write_buffer, raw_streams, output_idx, write_offset, end_output_idx);
        buffer <<= entry_nbits(entry);
        free_bits += entry_nbits(entry);
    }
    if (output_idx < end_output_idx) {
        buffer |= static_cast<uint64_t>(lookahead) << (free_bits - 32U);
    }
    while (output_idx < end_output_idx) {
        const Entry entry = decode_lut_entry(lut, buffer, max_bits);
        output_idx = write_entry<Word, Reorder, MantissaBits, ExponentBits, StreamId, StageSymbols>(
            entry, write_buffer, raw_streams, output_idx, write_offset, end_output_idx);
        buffer <<= entry_nbits(entry);
    }
    if (!StageSymbols) return;
    __syncthreads();

    const uint32_t block_output_count =
        block_end > write_offset ? block_end - write_offset : 0U;
    for (uint32_t index = THREAD_ID; index < block_output_count;
         index += N_THREADS) {
        outputs[index + write_offset] = assemble_value<Word, Reorder, MantissaBits, ExponentBits, StreamId>(
            reinterpret_cast<uint8_t*>(write_buffer)[index], raw_streams, index + write_offset);
    }
}

template <typename Word, bool Reorder, int MantissaBits, int ExponentBits, int StreamId, bool StageSymbols>
__device__ __forceinline__ void decode_long_pairs_block(
    const uint32_t* __restrict__ lut,
    const int max_bits,
    const uint8_t* __restrict__ codes,
    const uint8_t* const* raw_streams,
    const uint8_t* __restrict__ gaps,
    const uint32_t* __restrict__ output_positions,
    Word* __restrict__ outputs,
    const int n_bytes,
    const uint32_t local_block_id,
    uint8_t* __restrict__ shared_mem) {
    // Rare long codes may cross a segment by 55 bits; retain full lookahead.
    const uint64_t thread_id = static_cast<uint64_t>(local_block_id) * N_THREADS + THREAD_ID;
    const uint64_t code_offset = thread_id * BYTES_PER_THREAD;
    uint64_t high = 0, low = 0;
#pragma unroll
    for (int index = 0; index < 16; ++index) {
        const uint64_t source = code_offset + index;
        const uint64_t byte = source < static_cast<uint64_t>(n_bytes) ? codes[source] : 0U;
        if (index < 8) high |= byte << (56U - 8U * index);
        else low |= byte << (56U - 8U * (index - 8));
    }
    const unsigned gap = gaps[thread_id];
    unsigned consumed = code_offset < static_cast<uint64_t>(n_bytes) ? gap : 64U;
    uint32_t count = 0;
    while (consumed < 64U) {
        const uint32_t entry = decode_lut_entry(lut, peek_bits(high, low, consumed), max_bits);
        count += entry_count(entry);
        consumed += entry_nbits(entry);
    }
    const uint32_t start = __ldg(output_positions + local_block_id);
    const uint32_t block_end = __ldg(output_positions + local_block_id + 1U);
    uint32_t index = start + exclusive_block_prefix(count, reinterpret_cast<uint32_t*>(shared_mem));
    const uint32_t end = static_cast<uint32_t>(
        min(static_cast<uint64_t>(index) + count, static_cast<uint64_t>(block_end)));
    Word* write_buffer = StageSymbols ? reinterpret_cast<Word*>(
        shared_mem + ((N_THREADS + 31U) >> 5U) * sizeof(uint32_t)) : outputs + start;
    consumed = gap;
    while (index < end) {
        const uint32_t entry = decode_lut_entry(lut, peek_bits(high, low, consumed), max_bits);
        index = write_entry<Word, Reorder, MantissaBits, ExponentBits, StreamId, StageSymbols>(
            entry, write_buffer, raw_streams, index, start, end);
        consumed += entry_nbits(entry);
    }
    if (!StageSymbols) return;
    __syncthreads();
    for (uint32_t output = start + THREAD_ID; output < block_end; output += N_THREADS) {
        outputs[output] = assemble_value<Word, Reorder, MantissaBits, ExponentBits, StreamId>(
            reinterpret_cast<uint8_t*>(write_buffer)[output - start], raw_streams, output);
    }
}

}  // namespace

template <typename Word, bool Reorder, int MantissaBits, int ExponentBits, int StreamId, bool Mixed, bool LongCodes, bool StageSymbols>
__global__ void decode_batched(
    const uint64_t* __restrict__ lut_ptrs,
    const uint64_t* __restrict__ code_ptrs,
    const uint64_t* __restrict__ raw_stream_ptrs,
    const uint64_t* __restrict__ gap_ptrs,
    const uint64_t* __restrict__ output_position_ptrs,
    const uint64_t* __restrict__ output_ptrs,
    const int32_t* __restrict__ max_bits,
    const int32_t* __restrict__ n_bytes,
    const int32_t* __restrict__ block_group_ids,
    const int32_t* __restrict__ block_local_ids) {
    const int32_t group_id = __ldg(block_group_ids + BLOCK_ID);
    const int32_t local_block_id = __ldg(block_local_ids + BLOCK_ID);
    const int32_t group_n_bytes = __ldg(n_bytes + group_id);
    const int32_t group_max_bits = __ldg(max_bits + group_id);

    const uint64_t lut_address = __ldg(lut_ptrs + group_id);
    const uint64_t code_address = __ldg(code_ptrs + group_id);
    const uint8_t* raw_streams[sizeof(Word)] = {};
#pragma unroll
    for (int lane = 0; lane < sizeof(Word); ++lane) {
        if (lane != StreamId) {
            raw_streams[lane] = reinterpret_cast<const uint8_t*>(
                __ldg(raw_stream_ptrs + group_id * sizeof(Word) + lane));
        }
    }
    const uint64_t gap_address = __ldg(gap_ptrs + group_id);
    const uint64_t output_position_address = __ldg(
        output_position_ptrs + group_id);
    const uint64_t output_address = __ldg(output_ptrs + group_id);

    extern __shared__ __align__(16) uint8_t shared_mem[];
    const uint8_t* codes = reinterpret_cast<const uint8_t*>(code_address);
    const uint8_t* gaps = reinterpret_cast<const uint8_t*>(gap_address);
    const uint32_t* positions = reinterpret_cast<const uint32_t*>(output_position_address);
    Word* outputs = reinterpret_cast<Word*>(output_address);
    if (group_max_bits == 0) {
        const uint64_t start = static_cast<uint64_t>(local_block_id) * N_THREADS * BYTES_PER_THREAD;
        for (uint64_t index = start + THREAD_ID;
             index < start + N_THREADS * BYTES_PER_THREAD && index < group_n_bytes;
             index += N_THREADS) {
            outputs[index] = assemble_value<Word, Reorder, MantissaBits, ExponentBits, StreamId>(
                codes[index], raw_streams, index);
        }
    } else if (!Mixed) {
        decode_short_block<Word, Reorder, MantissaBits, ExponentBits, StreamId, false, StageSymbols>(
            reinterpret_cast<const uint16_t*>(lut_address), group_max_bits, codes,
            raw_streams, gaps, positions, outputs, group_n_bytes, local_block_id, shared_mem);
    } else if (!LongCodes) {
        decode_short_block<Word, Reorder, MantissaBits, ExponentBits, StreamId, true, StageSymbols>(
            reinterpret_cast<const uint32_t*>(lut_address), group_max_bits, codes,
            raw_streams, gaps, positions, outputs, group_n_bytes, local_block_id, shared_mem);
    } else {
        decode_long_pairs_block<Word, Reorder, MantissaBits, ExponentBits, StreamId, StageSymbols>(
            reinterpret_cast<const uint32_t*>(lut_address), group_max_bits, codes,
            raw_streams, gaps, positions, outputs, group_n_bytes, local_block_id, shared_mem);
    }
}
