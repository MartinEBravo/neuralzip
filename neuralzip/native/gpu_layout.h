constexpr unsigned kGPUCodeMaxBits = 12;
constexpr unsigned kGPUGapBits = 4;
constexpr unsigned kGPUGapLimit = 1U << kGPUGapBits;
constexpr std::size_t kGPUDecoderIndexLimit =
    static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
static_assert(
    kGPUCodeMaxBits <= kGPUGapLimit,
    "GPU restart gaps must represent every supported code length.");

struct GPUBFloat16Encoding {
    std::vector<std::uint8_t> encoded_exponents;
    std::vector<std::uint8_t> sign_mantissa;
    std::vector<std::uint8_t> packed_gaps;
    std::vector<std::uint32_t> output_positions;
    std::uint64_t bit_length = 0;
    unsigned max_bits = 0;
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

GPUBFloat16Encoding encode_gpu_bfloat16_layout(
    const std::uint8_t* raw,
    std::size_t raw_size,
    const std::uint8_t* codebook_header,
    Py_ssize_t codebook_header_size) {
    constexpr std::size_t bytes_per_thread = 8;
    constexpr std::size_t threads_per_block = 512;
    require(
        (raw_size & 1U) == 0,
        "A BF16 byte sequence must have an even length.");
    const std::size_t numel = raw_size / 2U;
    require<std::overflow_error>(
        numel <= kGPUDecoderIndexLimit,
        "BF16 element count exceeds the int32 CUDA decoder ABI limit.");

    const ExportedCodeTable table = export_code_table_from_header(
        codebook_header, codebook_header_size, kGPUCodeMaxBits);
    require(
        table.max_bits > 0 && table.max_bits <= kGPUCodeMaxBits,
        "GPU Huffman code length must be in [1, 12].");

    GPUBFloat16Encoding result;
    result.max_bits = table.max_bits;
    result.sign_mantissa.resize(numel);

    for (std::size_t index = 0; index < numel; ++index) {
        const std::uint8_t low = raw[index * 2U];
        const std::uint8_t high = raw[index * 2U + 1U];
        const std::uint8_t exponent = static_cast<std::uint8_t>(
            ((high & 0x7FU) << 1U) | (low >> 7U));
        const unsigned length = table.code_lengths[exponent];
        if (length == 0) {
            throw std::runtime_error(
                "Exponent symbol " + std::to_string(exponent)
                + " at index " + std::to_string(index)
                + " has no codebook entry.");
        }
        result.bit_length += length;
        result.sign_mantissa[index] = static_cast<std::uint8_t>(
            (high & 0x80U) | (low & 0x7FU));
    }

    const std::uint64_t encoded_bytes_u64 =
        result.bit_length / 8U + (result.bit_length % 8U != 0 ? 1U : 0U);
    require<std::overflow_error>(
        encoded_bytes_u64 <= kGPUDecoderIndexLimit,
        "Encoded exponent stream exceeds the int32 CUDA decoder ABI limit.");
    const std::size_t encoded_bytes = static_cast<std::size_t>(
        encoded_bytes_u64);
    result.encoded_exponents.resize(encoded_bytes);

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
        block_count + 1U, static_cast<std::uint32_t>(numel));

    const std::uint64_t segment_bits = bytes_per_thread * 8U;
    const std::uint64_t block_bits =
        static_cast<std::uint64_t>(block_bytes) * 8U;
    std::size_t next_segment = 0;
    std::size_t next_block = 0;
    std::size_t encoded_offset = 0;
    std::uint64_t symbol_start = 0;
    std::uint32_t accumulator = 0;
    unsigned pending = 0;

    for (std::size_t index = 0; index < numel; ++index) {
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

        const std::uint8_t low = raw[index * 2U];
        const std::uint8_t high = raw[index * 2U + 1U];
        const std::uint8_t exponent = static_cast<std::uint8_t>(
            ((high & 0x7FU) << 1U) | (low >> 7U));
        const unsigned length = table.code_lengths[exponent];
        accumulator = (accumulator << length) | table.code_values[exponent];
        pending += length;
        symbol_start += length;

        while (pending >= 8U) {
            pending -= 8U;
            result.encoded_exponents[encoded_offset++] =
                static_cast<std::uint8_t>((accumulator >> pending) & 0xFFU);
            accumulator = pending == 0
                ? 0U
                : accumulator & ((1U << pending) - 1U);
        }
    }
    if (pending != 0) {
        result.encoded_exponents[encoded_offset++] =
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
