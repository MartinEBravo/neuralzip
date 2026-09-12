constexpr unsigned kGPUCodeMaxBits = 12;
constexpr unsigned kGPUGapBits = 4;
constexpr unsigned kGPUGapLimit = 1U << kGPUGapBits;
constexpr std::size_t kGPUDecoderIndexLimit =
    static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
static_assert(
    kGPUCodeMaxBits <= kGPUGapLimit,
    "GPU restart gaps must represent every supported code length.");

struct GPUHuffmanEncoding {
    std::vector<std::uint8_t> bitstream;
    std::vector<std::uint8_t> packed_gaps;
    std::vector<std::uint32_t> output_positions;
    std::uint64_t bit_length = 0;
    unsigned max_bits = 0;
};

struct GPULayoutEncoding {
    GPUHuffmanEncoding symbols;
    std::vector<std::vector<std::uint8_t>> raw_streams;
};

std::size_t checked_size_product(
    std::size_t left,
    std::size_t right,
    const char* message) {
    require<std::overflow_error>(
        left == 0 || right <= std::numeric_limits<std::size_t>::max() / left,
        message);
    return left * right;
}

std::size_t checked_ceil_div(
    std::size_t numerator,
    std::size_t denominator) {
    require(denominator > 0, "GPU layout divisor must be positive.");
    return numerator / denominator + (numerator % denominator != 0 ? 1U : 0U);
}

std::vector<std::uint8_t> pack_gpu_gaps(
    const std::vector<std::uint8_t>& gaps) {
    std::vector<std::uint8_t> output(
        gaps.size() / 2U + (gaps.size() % 2U), 0);
    for (std::size_t index = 0; index < gaps.size(); ++index) {
        const std::uint8_t gap = gaps[index];
        const unsigned shift = (index & 1U) == 0 ? kGPUGapBits : 0U;
        output[index / 2U] |= static_cast<std::uint8_t>(gap << shift);
    }
    return output;
}

// Both passes read identical symbols; only the first may save residual bytes.
template <typename SymbolReader>
GPUHuffmanEncoding encode_gpu_symbols(
    std::size_t symbol_count,
    const ExportedCodeTable& table,
    SymbolReader read_symbol) {
    constexpr std::size_t bytes_per_thread = 8;
    constexpr std::size_t threads_per_block = 512;
    require<std::overflow_error>(
        symbol_count <= kGPUDecoderIndexLimit,
        "Symbol count exceeds the int32 CUDA decoder ABI limit.");
    require(
        table.max_bits > 0 && table.max_bits <= kGPUCodeMaxBits,
        "GPU Huffman code length must be in [1, 12].");

    GPUHuffmanEncoding result;
    result.max_bits = table.max_bits;

    for (std::size_t index = 0; index < symbol_count; ++index) {
        const std::uint8_t symbol = read_symbol(index, true);
        const unsigned length = table.code_lengths[symbol];
        if (length == 0) {
            throw std::runtime_error(
                "Symbol " + std::to_string(symbol)
                + " at index " + std::to_string(index)
                + " has no codebook entry.");
        }
        result.bit_length += length;
    }

    const std::uint64_t encoded_bytes_u64 =
        result.bit_length / 8U + (result.bit_length % 8U != 0 ? 1U : 0U);
    require<std::overflow_error>(
        encoded_bytes_u64 <= kGPUDecoderIndexLimit,
        "Encoded symbol stream exceeds the int32 CUDA decoder ABI limit.");
    const std::size_t encoded_bytes = static_cast<std::size_t>(
        encoded_bytes_u64);
    result.bitstream.resize(encoded_bytes);

    const std::size_t block_bytes = checked_size_product(
        bytes_per_thread,
        threads_per_block,
        "GPU block byte size overflowed.");
    const std::size_t block_count = checked_ceil_div(
        encoded_bytes, block_bytes);
    const std::size_t gap_count = checked_size_product(
        block_count,
        threads_per_block,
        "GPU gap count overflowed.");
    const std::size_t populated_segments = checked_ceil_div(
        encoded_bytes, bytes_per_thread);
    std::vector<std::uint8_t> gaps(gap_count, 0);
    result.output_positions.assign(
        block_count + 1U, static_cast<std::uint32_t>(symbol_count));

    const std::uint64_t segment_bits = bytes_per_thread * 8U;
    const std::uint64_t block_bits =
        static_cast<std::uint64_t>(block_bytes) * 8U;
    std::size_t next_segment = 0;
    std::size_t next_block = 0;
    std::size_t encoded_offset = 0;
    std::uint64_t symbol_start = 0;
    std::uint32_t accumulator = 0;
    unsigned pending = 0;

    for (std::size_t index = 0; index < symbol_count; ++index) {
        while (
            next_segment < populated_segments
            && symbol_start
                >= static_cast<std::uint64_t>(next_segment) * segment_bits) {
            const std::uint64_t boundary =
                static_cast<std::uint64_t>(next_segment) * segment_bits;
            const std::uint64_t gap = symbol_start - boundary;
            if (gap >= kGPUGapLimit) {
                throw std::runtime_error(
                    "A code crosses too much of a CUDA segment to fit in "
                    "4-bit gaps.");
            }
            gaps[next_segment] = static_cast<std::uint8_t>(gap);
            ++next_segment;
        }
        while (
            next_block < block_count
            && symbol_start
                >= static_cast<std::uint64_t>(next_block) * block_bits) {
            result.output_positions[next_block] =
                static_cast<std::uint32_t>(index);
            ++next_block;
        }

        const std::uint8_t symbol = read_symbol(index, false);
        const unsigned length = table.code_lengths[symbol];
        accumulator = (accumulator << length) | table.code_values[symbol];
        pending += length;
        symbol_start += length;

        while (pending >= 8U) {
            pending -= 8U;
            result.bitstream[encoded_offset++] =
                static_cast<std::uint8_t>((accumulator >> pending) & 0xFFU);
            accumulator = pending == 0
                ? 0U
                : accumulator & ((1U << pending) - 1U);
        }
    }
    if (pending != 0) {
        result.bitstream[encoded_offset++] =
            static_cast<std::uint8_t>(
                (accumulator << (8U - pending)) & 0xFFU);
    }
    while (next_segment < populated_segments) {
        const std::uint64_t boundary =
            static_cast<std::uint64_t>(next_segment) * segment_bits;
        gaps[next_segment] = static_cast<std::uint8_t>(
            result.bit_length - boundary);
        ++next_segment;
    }
    result.packed_gaps = pack_gpu_gaps(gaps);
    return result;
}

template <unsigned ElementSize, bool ByteAlignedExponent>
GPULayoutEncoding encode_gpu_layout(
    const std::uint8_t* raw,
    std::size_t numel,
    const ExportedCodeTable& table,
    unsigned stream_id,
    bool bit_reorder,
    unsigned mantissa_bits,
    unsigned exponent_bits) {
    if constexpr (ByteAlignedExponent) {
        stream_id = ElementSize - 1U;
        bit_reorder = true;
        mantissa_bits = ElementSize * 8U - 9U;
        exponent_bits = 8;
    }
    const std::uint32_t mantissa_mask = (1U << mantissa_bits) - 1U;
    const std::uint32_t sign_mask = 1U << mantissa_bits;
    const std::uint32_t exponent_mask =
        ((1U << exponent_bits) - 1U) << (mantissa_bits + 1U);
    GPULayoutEncoding result;
    result.raw_streams.resize(ElementSize);
    for (unsigned lane = 0; lane < ElementSize; ++lane) {
        if (lane != stream_id) {
            result.raw_streams[lane].resize(numel);
        }
    }
    result.symbols = encode_gpu_symbols(
        numel, table, [&](std::size_t index, bool first_pass) {
            std::uint32_t value = 0;
            for (unsigned lane = 0; lane < ElementSize; ++lane) {
                value |= static_cast<std::uint32_t>(raw[index * ElementSize + lane])
                    << (8U * lane);
            }
            if (bit_reorder) {
                value = (value & mantissa_mask)
                    | ((value >> exponent_bits) & sign_mask)
                    | ((value << 1U) & exponent_mask);
            }
            if (first_pass) {
                for (unsigned lane = 0; lane < ElementSize; ++lane) {
                    if (lane != stream_id) {
                        result.raw_streams[lane][index] = static_cast<std::uint8_t>(
                            value >> (8U * lane));
                    }
                }
            }
            return static_cast<std::uint8_t>(value >> (8U * stream_id));
        });
    return result;
}
