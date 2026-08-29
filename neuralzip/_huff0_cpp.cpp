#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

extern "C" {
#include "huf.h"
#include "zstd_internal.h"
}

namespace {

namespace nb = nanobind;

class ReadOnlyBuffer {
public:
    ReadOnlyBuffer() = default;

    explicit ReadOnlyBuffer(nb::handle source) {
        acquire(source);
    }

    ~ReadOnlyBuffer() {
        release();
    }

    ReadOnlyBuffer(const ReadOnlyBuffer&) = delete;
    ReadOnlyBuffer& operator=(const ReadOnlyBuffer&) = delete;

    ReadOnlyBuffer(ReadOnlyBuffer&& other) noexcept : view_(other.view_) {
        other.view_ = {};
    }

    ReadOnlyBuffer& operator=(ReadOnlyBuffer&& other) noexcept {
        if (this != &other) {
            release();
            view_ = other.view_;
            other.view_ = {};
        }
        return *this;
    }

    void acquire(nb::handle source) {
        release();
        if (PyObject_GetBuffer(source.ptr(), &view_, PyBUF_CONTIG_RO) < 0) {
            throw nb::python_error();
        }
    }

    const std::uint8_t* data() const {
        return static_cast<const std::uint8_t*>(view_.buf);
    }

    Py_ssize_t size() const {
        return view_.len;
    }

    bool readonly() const {
        return view_.readonly != 0;
    }

private:
    void release() noexcept {
        if (view_.obj != nullptr) {
            PyBuffer_Release(&view_);
            view_ = {};
        }
    }

    Py_buffer view_{};
};

constexpr std::size_t kMaxBlockSize = HUF_BLOCKSIZE_MAX;
constexpr unsigned kMaxSymbolValue = HUF_SYMBOLVALUE_MAX;
constexpr unsigned kTableLog = HUF_TABLELOG_DEFAULT;
constexpr unsigned kMaxTableLog = HUF_TABLELOG_ABSOLUTEMAX;
constexpr std::uint64_t kMaxHistogramTotal =
    (static_cast<std::uint64_t>(1) << 30U) - 1U;

bool detect_runtime_bmi2() noexcept {
#if DYNAMIC_BMI2
    return ZSTD_cpuSupportsBmi2() != 0;
#else
    return false;
#endif
}

const bool kRuntimeBmi2 = detect_runtime_bmi2();
const int kFlags = HUF_flags_disableAsm |
                   (kRuntimeBmi2 ? HUF_flags_bmi2 : 0);

struct Huff0Codebook {
    std::vector<HUF_CElt> ctable;
    std::vector<std::uint8_t> serialized;
    bool covers_all_symbols = false;
};

struct Huff0Decodebook {
    std::vector<HUF_DTable> dtable_x1;
    std::vector<HUF_DTable> dtable_x2;
};

struct NGramBlockResult {
    std::vector<std::uint8_t> payload;
    std::vector<std::uint8_t> escapes;
    std::vector<std::uint8_t> tail;
    Py_ssize_t symbol_count = 0;
    bool encoded = false;
};

enum class MixedMode {
    Raw,
    SharedHuff0,
    PackedHuff0,
};

struct NGramVocabLookup {
    std::vector<std::uint64_t> keys;
    // Token 255 is reserved for escapes, so it is also a safe empty-slot
    // sentinel: valid vocabulary ids are always in [0, 254].
    std::vector<std::uint8_t> tokens;
    std::size_t mask = 0;
};

struct MixedStreamSpec {
    MixedMode mode = MixedMode::Raw;
    Huff0Codebook* codebook = nullptr;
    nb::object codebook_owner;
    std::vector<std::uint64_t> vocab;
    NGramVocabLookup vocab_lookup;
    int ngram_n = 1;
    std::size_t chunk_size = 0;
};

struct MixedBlockResult {
    MixedMode mode = MixedMode::Raw;
    std::vector<std::uint8_t> raw;
    std::vector<std::uint8_t> payload;
    std::vector<std::uint8_t> escapes;
    std::vector<std::uint8_t> tail;
    Py_ssize_t symbol_count = 0;
    Py_ssize_t token_count = 0;
    int ngram_n = 1;
    std::string error;
};

enum class MixedStreamDecodeMode {
    Raw,
    SharedHuff0,
};

struct MixedStreamDecodeChunk {
    MixedStreamDecodeMode mode = MixedStreamDecodeMode::Raw;
    ReadOnlyBuffer payload;
    nb::object decodebook_owner;
    Huff0Decodebook* decodebook = nullptr;
    std::size_t tensor_index = 0;
    int stream = 0;
    std::size_t output_offset = 0;
    std::size_t symbol_count = 0;
    std::string error;
};

struct MixedStreamDecodeTensor {
    int element_size = 0;
    bool bit_reorder = false;
    std::size_t numel = 0;
    std::vector<std::vector<std::uint8_t>> lanes;
    nb::object output;
    std::uint8_t* output_data = nullptr;
};

struct MixedStreamAlignedDecodeJob {
    std::size_t tensor_index = 0;
    std::size_t output_offset = 0;
    std::size_t symbol_count = 0;
    std::array<std::size_t, 4> chunk_indices{};
    std::string error;
};

struct MixedStreamCombineJob {
    std::size_t tensor_index = 0;
    std::size_t start = 0;
    std::size_t length = 0;
};

template <typename Worker>
std::string run_parallel_jobs(
    std::size_t job_count,
    std::size_t total_input_bytes,
    int requested_threads,
    Worker&& process_job) {
    if (job_count == 0) {
        return {};
    }
    unsigned hardware_threads = std::thread::hardware_concurrency();
    int default_threads = static_cast<int>(hardware_threads ? hardware_threads : 1U);
    int worker_count = requested_threads > 0 ? requested_threads : default_threads;
    worker_count = std::max(1, std::min({worker_count, 16, static_cast<int>(job_count)}));
    if (total_input_bytes < (512U << 10)) {
        worker_count = 1;
    }

    std::atomic<std::size_t> next{0};
    auto worker = [&]() {
        while (true) {
            std::size_t index = next.fetch_add(1, std::memory_order_relaxed);
            if (index >= job_count) {
                return;
            }
            process_job(index);
        }
    };

    if (worker_count == 1) {
        worker();
        return {};
    }

    std::string thread_error;
    std::vector<std::thread> workers;
    try {
        workers.reserve(static_cast<std::size_t>(worker_count));
        for (int index = 0; index < worker_count; ++index) {
            workers.emplace_back(worker);
        }
    } catch (const std::exception& error) {
        thread_error = error.what();
    }
    for (auto& thread : workers) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    return thread_error;
}

void check_huf(size_t code) {
    if (HUF_isError(code)) {
        throw std::runtime_error(HUF_getErrorName(code));
    }
}

nb::bytes bytes_from_vector(const std::vector<std::uint8_t>& data) {
    return nb::bytes(data.data(), data.size());
}

nb::tuple ngram_result_to_tuple(const NGramBlockResult& result) {
    return nb::make_tuple(
        bytes_from_vector(result.payload),
        result.symbol_count,
        bytes_from_vector(result.escapes),
        bytes_from_vector(result.tail));
}

inline std::uint8_t byte_group_stream_byte(
    const std::uint8_t* raw,
    int element_size,
    bool bit_reorder,
    int stream,
    std::size_t position) {
    std::size_t source = position * static_cast<std::size_t>(element_size);
    if (!bit_reorder || stream < element_size - 2) {
        return raw[source + static_cast<std::size_t>(stream)];
    }
    if (stream == element_size - 2) {
        return static_cast<std::uint8_t>(
            (raw[source + static_cast<std::size_t>(stream)] & 0x7FU)
            | (raw[source + static_cast<std::size_t>(element_size - 1)] & 0x80U));
    }
    return static_cast<std::uint8_t>(
        ((raw[source + static_cast<std::size_t>(element_size - 1)] << 1U) & 0xFEU)
        | (raw[source + static_cast<std::size_t>(element_size - 2)] >> 7U));
}

void extract_byte_group_stream_block(
    std::vector<std::uint8_t>& output,
    const std::uint8_t* raw,
    int element_size,
    bool bit_reorder,
    int stream,
    std::size_t start,
    std::size_t length) {
    output.resize(length);
    for (std::size_t offset = 0; offset < length; ++offset) {
        output[offset] = byte_group_stream_byte(
            raw, element_size, bit_reorder, stream, start + offset);
    }
}

void restore_mixed_stream_chunk(
    const MixedStreamDecodeTensor& request,
    const std::array<const std::uint8_t*, 4>& lanes,
    std::size_t output_offset,
    std::size_t length) {
    if (length == 0) {
        return;
    }
    const int element_size = request.element_size;
    std::uint8_t* output = request.output_data
        + output_offset * static_cast<std::size_t>(element_size);
    if (element_size == 2) {
        const std::uint8_t* first = lanes[0];
        const std::uint8_t* second = lanes[1];
        if (!request.bit_reorder) {
            for (std::size_t index = 0; index < length; ++index) {
                const std::size_t target = index * 2U;
                output[target] = first[index];
                output[target + 1U] = second[index];
            }
            return;
        }
        for (std::size_t index = 0; index < length; ++index) {
            const std::uint8_t penultimate = first[index];
            const std::uint8_t last = second[index];
            const std::size_t target = index * 2U;
            output[target] = static_cast<std::uint8_t>(
                (penultimate & 0x7FU) | ((last & 0x01U) << 7U));
            output[target + 1U] = static_cast<std::uint8_t>(
                (last >> 1U) | (penultimate & 0x80U));
        }
        return;
    }

    const std::uint8_t* first = lanes[0];
    const std::uint8_t* second = lanes[1];
    const std::uint8_t* penultimate_lane = lanes[2];
    const std::uint8_t* last_lane = lanes[3];
    if (!request.bit_reorder) {
        for (std::size_t index = 0; index < length; ++index) {
            const std::size_t target = index * 4U;
            output[target] = first[index];
            output[target + 1U] = second[index];
            output[target + 2U] = penultimate_lane[index];
            output[target + 3U] = last_lane[index];
        }
        return;
    }
    for (std::size_t index = 0; index < length; ++index) {
        const std::uint8_t penultimate = penultimate_lane[index];
        const std::uint8_t last = last_lane[index];
        const std::size_t target = index * 4U;
        output[target] = first[index];
        output[target + 1U] = second[index];
        output[target + 2U] = static_cast<std::uint8_t>(
            (penultimate & 0x7FU) | ((last & 0x01U) << 7U));
        output[target + 3U] = static_cast<std::uint8_t>(
            (last >> 1U) | (penultimate & 0x80U));
    }
}

void restore_mixed_stream_tensor_range(
    const MixedStreamDecodeTensor& request,
    std::size_t start,
    std::size_t length) {
    if (length == 0) {
        return;
    }
    std::array<const std::uint8_t*, 4> lanes{};
    for (int stream = 0; stream < request.element_size; ++stream) {
        lanes[static_cast<std::size_t>(stream)] = request.lanes[
            static_cast<std::size_t>(stream)].data() + start;
    }
    restore_mixed_stream_chunk(request, lanes, start, length);
}

void release_vector(std::vector<std::uint8_t>& data) {
    std::vector<std::uint8_t>().swap(data);
}

std::array<unsigned, 256> count_symbols(const std::uint8_t* data, Py_ssize_t data_len, unsigned prior_count) {
    std::array<unsigned, 256> counts{};
    if (prior_count > 0) {
        counts.fill(prior_count);
    }
    for (Py_ssize_t index = 0; index < data_len; ++index) {
        if (counts[data[index]] == std::numeric_limits<unsigned>::max()) {
            throw std::runtime_error("Huff0 symbol count overflow.");
        }
        counts[data[index]] += 1;
    }
    return counts;
}

unsigned nonzero_symbol_count(const std::array<unsigned, 256>& counts) {
    unsigned nonzero = 0;
    for (unsigned count : counts) {
        nonzero += count > 0 ? 1U : 0U;
    }
    return nonzero;
}

unsigned max_observed_symbol(const std::array<unsigned, 256>& counts) {
    for (int symbol = 255; symbol >= 0; --symbol) {
        if (counts[static_cast<std::size_t>(symbol)] > 0) {
            return static_cast<unsigned>(symbol);
        }
    }
    return 0;
}

std::unique_ptr<Huff0Codebook> make_codebook_from_counts(
    const std::array<unsigned, 256>& counts,
    unsigned max_bits) {
    const unsigned symbol_count = nonzero_symbol_count(counts);
    if (symbol_count < 2) {
        throw std::runtime_error("Huff0 requires at least two symbols.");
    }
    if (symbol_count > (1U << max_bits)) {
        throw std::runtime_error(
            "Huff0 max_bits cannot represent every observed symbol.");
    }
    const unsigned max_symbol_value = max_observed_symbol(counts);

    auto codebook = std::make_unique<Huff0Codebook>();
    codebook->ctable.resize(HUF_CTABLE_SIZE_ST(kMaxSymbolValue));
    codebook->covers_all_symbols = symbol_count == counts.size();
    std::vector<unsigned> build_workspace(HUF_CTABLE_WORKSPACE_SIZE_U32);
    const size_t table_log = HUF_buildCTable_wksp(
        codebook->ctable.data(),
        counts.data(),
        max_symbol_value,
        max_bits,
        build_workspace.data(),
        build_workspace.size() * sizeof(unsigned));
    check_huf(table_log);

    codebook->serialized.resize(HUF_CTABLEBOUND);
    std::vector<std::uint64_t> write_workspace(HUF_WORKSPACE_SIZE_U64);
    const size_t header_size = HUF_writeCTable_wksp(
        codebook->serialized.data(),
        codebook->serialized.size(),
        codebook->ctable.data(),
        max_symbol_value,
        static_cast<unsigned>(table_log),
        write_workspace.data(),
        write_workspace.size() * sizeof(std::uint64_t));
    check_huf(header_size);
    codebook->serialized.resize(header_size);
    return codebook;
}

std::unique_ptr<Huff0Codebook> make_codebook(
    const std::uint8_t* data,
    Py_ssize_t data_len,
    unsigned prior_count,
    unsigned max_bits) {
    if (data_len <= 0) {
        throw std::runtime_error("Cannot build a Huff0 codebook from empty data.");
    }

    return make_codebook_from_counts(
        count_symbols(data, data_len, prior_count), max_bits);
}

std::unique_ptr<Huff0Codebook> make_codebook_from_header(
    const std::uint8_t* header,
    Py_ssize_t header_len) {
    if (header_len <= 0) {
        throw std::runtime_error("Huff0 codebook header is empty.");
    }

    auto codebook = std::make_unique<Huff0Codebook>();
    codebook->ctable.resize(HUF_CTABLE_SIZE_ST(kMaxSymbolValue));
    unsigned max_symbol_value = kMaxSymbolValue;
    unsigned has_zero_weights = 0;
    const size_t read_size = HUF_readCTable(
        codebook->ctable.data(),
        &max_symbol_value,
        header,
        static_cast<std::size_t>(header_len),
        &has_zero_weights);
    check_huf(read_size);
    if (read_size != static_cast<std::size_t>(header_len)) {
        throw std::runtime_error(
            "Huff0 did not consume the full codebook header.");
    }
    codebook->covers_all_symbols =
        max_symbol_value == kMaxSymbolValue && has_zero_weights == 0;
    codebook->serialized.assign(header, header + header_len);
    return codebook;
}

struct ExportedCodeTable {
    std::array<std::uint8_t, kMaxSymbolValue + 1> code_lengths{};
    std::array<std::uint16_t, kMaxSymbolValue + 1> code_values{};
    unsigned max_bits = 0;
};

ExportedCodeTable export_code_table_from_header(
    const std::uint8_t* header,
    Py_ssize_t header_len,
    unsigned max_bits) {
    auto codebook = make_codebook_from_header(header, header_len);
    if (max_bits == 0 || max_bits > HUF_TABLELOG_ABSOLUTEMAX) {
        throw std::runtime_error(
            "max_bits must be in [1, HUF_TABLELOG_ABSOLUTEMAX].");
    }

    const HUF_CTableHeader table_header = HUF_readCTableHeader(
        codebook->ctable.data());
    if (table_header.tableLog > max_bits) {
        throw std::runtime_error(
            "Huff0 codebook exceeds the requested max_bits limit.");
    }

    constexpr unsigned ctable_word_bits = sizeof(HUF_CElt) * 8U;
    ExportedCodeTable output;
    for (unsigned symbol = 0; symbol <= kMaxSymbolValue; ++symbol) {
        const unsigned length = HUF_getNbBitsFromCTable(
            codebook->ctable.data(), symbol);
        if (length > max_bits || length > HUF_TABLELOG_ABSOLUTEMAX) {
            throw std::runtime_error(
                "Huff0 symbol code exceeds the supported bit limit.");
        }
        output.code_lengths[symbol] = static_cast<std::uint8_t>(length);
        output.max_bits = std::max(output.max_bits, length);
        if (length == 0) {
            continue;
        }
        const HUF_CElt entry = codebook->ctable[
            static_cast<std::size_t>(symbol) + 1U];
        const HUF_CElt value = entry >> (ctable_word_bits - length);
        if (value >= (static_cast<HUF_CElt>(1U) << length)) {
            throw std::runtime_error("Invalid Huff0 code value.");
        }
        output.code_values[symbol] = static_cast<std::uint16_t>(value);
    }
    return output;
}

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
    if (
        left != 0
        && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error(message);
    }
    return left * right;
}

std::size_t checked_ceil_div(
    std::size_t numerator,
    std::size_t denominator) {
    if (denominator == 0) {
        throw std::runtime_error("GPU layout divisor must be positive.");
    }
    return numerator / denominator + (numerator % denominator != 0 ? 1U : 0U);
}

std::vector<std::uint8_t> pack_gpu_gaps(
    const std::vector<std::uint8_t>& gaps) {
    const std::size_t packed_bits = checked_size_product(
        gaps.size(), kGPUGapBits, "Packed GPU gap size overflowed.");
    const std::size_t packed_bytes = checked_ceil_div(packed_bits, 8U);
    std::vector<std::uint8_t> output;
    output.reserve(packed_bytes);

    std::uint32_t accumulator = 0;
    unsigned pending = 0;
    for (const std::uint8_t gap : gaps) {
        if (gap >= kGPUGapLimit) {
            throw std::runtime_error("GPU restart gap does not fit in 4 bits.");
        }
        accumulator = (accumulator << kGPUGapBits) | gap;
        pending += kGPUGapBits;
        while (pending >= 8U) {
            pending -= 8U;
            output.push_back(static_cast<std::uint8_t>(
                (accumulator >> pending) & 0xFFU));
            accumulator = pending == 0
                ? 0U
                : accumulator & ((1U << pending) - 1U);
        }
    }
    if (pending != 0) {
        output.push_back(static_cast<std::uint8_t>(
            (accumulator << (8U - pending)) & 0xFFU));
    }
    if (output.size() != packed_bytes) {
        throw std::runtime_error("Packed GPU gap size mismatch.");
    }
    return output;
}

GPUBFloat16Encoding encode_gpu_bfloat16_layout(
    const std::uint8_t* raw,
    std::size_t raw_size,
    const std::uint8_t* codebook_header,
    Py_ssize_t codebook_header_size,
    std::uint32_t bytes_per_thread,
    std::uint32_t threads_per_block) {
    if ((raw_size & 1U) != 0) {
        throw std::runtime_error(
            "A BF16 byte sequence must have an even length.");
    }
    const std::size_t numel = raw_size / 2U;
    if (numel > kGPUDecoderIndexLimit) {
        throw std::overflow_error(
            "BF16 element count exceeds the int32 CUDA decoder ABI limit.");
    }

    const ExportedCodeTable table = export_code_table_from_header(
        codebook_header, codebook_header_size, kGPUCodeMaxBits);
    if (table.max_bits == 0 || table.max_bits > kGPUCodeMaxBits) {
        throw std::runtime_error("GPU Huffman code length must be in [1, 12].");
    }

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
        if (
            result.bit_length
            > std::numeric_limits<std::uint64_t>::max() - length) {
            throw std::overflow_error("GPU exponent bit length overflowed.");
        }
        result.bit_length += length;
        result.sign_mantissa[index] = static_cast<std::uint8_t>(
            (high & 0x80U) | (low & 0x7FU));
    }

    const std::uint64_t encoded_bytes_u64 =
        result.bit_length / 8U + (result.bit_length % 8U != 0 ? 1U : 0U);
    if (encoded_bytes_u64 > kGPUDecoderIndexLimit) {
        throw std::overflow_error(
            "Encoded exponent stream exceeds the int32 CUDA decoder ABI limit.");
    }
    const std::size_t encoded_bytes = static_cast<std::size_t>(
        encoded_bytes_u64);
    result.encoded_exponents.resize(encoded_bytes);

    const std::size_t block_bytes = checked_size_product(
        static_cast<std::size_t>(bytes_per_thread),
        static_cast<std::size_t>(threads_per_block),
        "GPU block byte size overflowed.");
    if (block_bytes > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error(
            "GPU block byte size exceeds the uint32 layout limit.");
    }
    const std::size_t block_count = checked_ceil_div(
        encoded_bytes, block_bytes);
    if (block_count > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("GPU block count exceeds uint32.");
    }
    const std::size_t gap_count = checked_size_product(
        block_count,
        static_cast<std::size_t>(threads_per_block),
        "GPU gap count overflowed.");
    const std::size_t populated_segments = checked_ceil_div(
        encoded_bytes, static_cast<std::size_t>(bytes_per_thread));
    std::vector<std::uint8_t> gaps(gap_count, 0);
    result.output_positions.assign(
        block_count + 1U, static_cast<std::uint32_t>(numel));

    const std::uint64_t segment_bits =
        static_cast<std::uint64_t>(bytes_per_thread) * 8U;
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
    if (
        symbol_start != result.bit_length
        || encoded_offset != result.encoded_exponents.size()) {
        throw std::runtime_error("GPU exponent stream size mismatch.");
    }

    while (next_segment < populated_segments) {
        const std::uint64_t boundary =
            static_cast<std::uint64_t>(next_segment) * segment_bits;
        if (
            result.bit_length < boundary
            || result.bit_length - boundary >= kGPUGapLimit) {
            throw std::runtime_error(
                "A final code crosses too much of a CUDA segment to fit in "
                "4-bit gaps.");
        }
        gaps[next_segment] = static_cast<std::uint8_t>(
            result.bit_length - boundary);
        ++next_segment;
    }
    while (next_block < block_count) {
        result.output_positions[next_block] =
            static_cast<std::uint32_t>(numel);
        ++next_block;
    }
    result.output_positions[block_count] = static_cast<std::uint32_t>(numel);
    result.packed_gaps = pack_gpu_gaps(gaps);
    return result;
}

std::vector<std::uint8_t> encode_huff0_block(
    const std::uint8_t* data,
    Py_ssize_t data_len,
    const Huff0Codebook& codebook,
    std::size_t max_output_bytes) {
    if (data_len <= 0 || static_cast<std::size_t>(data_len) > kMaxBlockSize) {
        return {};
    }
    if (!codebook.covers_all_symbols) {
        for (Py_ssize_t index = 0; index < data_len; ++index) {
            if (HUF_getNbBitsFromCTable(codebook.ctable.data(), data[index]) == 0) {
                return {};
            }
        }
    }
    const std::size_t capacity = HUF_compressBound(
        static_cast<std::size_t>(data_len));
    if (capacity == 0) {
        return {};
    }
    std::vector<std::uint8_t> output(capacity);
    const size_t size = HUF_compress4X_usingCTable(
        output.data(), output.size(), data,
        static_cast<std::size_t>(data_len), codebook.ctable.data(), kFlags);
    if (HUF_isError(size)) {
        if (max_output_bytes != std::numeric_limits<std::size_t>::max()) {
            return {};
        }
        check_huf(size);
    }
    if (size == 0 || size > max_output_bytes) {
        return {};
    }
    output.resize(size);
    return output;
}

std::unique_ptr<Huff0Decodebook> make_decodebook_from_header(const std::uint8_t* header, Py_ssize_t header_len) {
    if (header_len <= 0) {
        throw std::runtime_error("Huff0 codebook header is empty.");
    }

    auto decodebook = std::make_unique<Huff0Decodebook>();
    decodebook->dtable_x1.resize(HUF_DTABLE_SIZE(HUF_TABLELOG_MAX - 1));
    decodebook->dtable_x1[0] = static_cast<HUF_DTable>(
        (HUF_TABLELOG_MAX - 1) * 0x01000001U);
    decodebook->dtable_x2.resize(HUF_DTABLE_SIZE(HUF_TABLELOG_MAX));
    decodebook->dtable_x2[0] = static_cast<HUF_DTable>(
        HUF_TABLELOG_MAX * 0x01000001U);
    std::vector<unsigned> workspace(HUF_DECOMPRESS_WORKSPACE_SIZE_U32);
    const size_t read_size_x1 = HUF_readDTableX1_wksp(
        decodebook->dtable_x1.data(),
        header,
        static_cast<std::size_t>(header_len),
        workspace.data(),
        workspace.size() * sizeof(unsigned),
        kFlags);
    check_huf(read_size_x1);
    if (read_size_x1 != static_cast<std::size_t>(header_len)) {
        throw std::runtime_error(
            "Huff0 X1 did not consume the full codebook header.");
    }
    const size_t read_size_x2 = HUF_readDTableX2_wksp(
        decodebook->dtable_x2.data(),
        header,
        static_cast<std::size_t>(header_len),
        workspace.data(),
        workspace.size() * sizeof(unsigned),
        kFlags);
    check_huf(read_size_x2);
    if (read_size_x2 != static_cast<std::size_t>(header_len)) {
        throw std::runtime_error(
            "Huff0 X2 did not consume the full codebook header.");
    }
    return decodebook;
}

const HUF_DTable* select_decode_dtable(
    const Huff0Decodebook& decodebook,
    std::size_t symbol_count,
    std::size_t payload_len) {
    return HUF_selectDecoder(symbol_count, payload_len) != 0
        ? decodebook.dtable_x2.data()
        : decodebook.dtable_x1.data();
}

void decode_huff0_block_into(
    const std::uint8_t* payload,
    Py_ssize_t payload_len,
    std::uint8_t* output,
    std::size_t symbol_count,
    const Huff0Decodebook& decodebook) {
    if (payload_len < 0) {
        throw std::runtime_error("payload_len must be non-negative.");
    }
    if (symbol_count == 0) {
        return;
    }
    const HUF_DTable* dtable = select_decode_dtable(
        decodebook, symbol_count, static_cast<std::size_t>(payload_len));
    const size_t decoded = HUF_decompress4X_usingDTable(
        output, symbol_count, payload,
        static_cast<std::size_t>(payload_len), dtable, kFlags);
    check_huf(decoded);
    if (decoded != symbol_count) {
        throw std::runtime_error("Huff0 decoded size mismatch.");
    }
}

std::vector<std::uint8_t> decode_huff0_block(
    const std::uint8_t* payload,
    Py_ssize_t payload_len,
    Py_ssize_t symbol_count,
    const Huff0Decodebook& decodebook) {
    if (symbol_count < 0) {
        throw std::runtime_error("symbol_count must be non-negative.");
    }
    std::vector<std::uint8_t> output(static_cast<std::size_t>(symbol_count));
    decode_huff0_block_into(
        payload, payload_len, output.data(), output.size(), decodebook);
    return output;
}

std::vector<std::uint8_t> restore_ngram_tokens(
    const std::vector<std::uint8_t>& tokens,
    int n,
    const std::vector<std::uint64_t>& vocab,
    const std::uint8_t* escapes,
    Py_ssize_t escape_len,
    const std::uint8_t* tail,
    Py_ssize_t tail_len) {
    if (n < 2 || n > 8) {
        throw std::runtime_error("n must be in [2, 8].");
    }
    if (tail_len < 0 || tail_len >= n) {
        throw std::runtime_error("Packed n-gram tail length is invalid.");
    }

    std::size_t escape_offset = 0;
    std::vector<std::uint8_t> output;
    output.reserve(tokens.size() * static_cast<std::size_t>(n)
                   + static_cast<std::size_t>(tail_len));
    for (std::uint8_t token : tokens) {
        if (token == 255) {
            std::size_t end = escape_offset + static_cast<std::size_t>(n);
            if (end > static_cast<std::size_t>(escape_len)) {
                throw std::runtime_error("Packed n-gram escape stream is truncated.");
            }
            output.insert(output.end(), escapes + escape_offset, escapes + end);
            escape_offset = end;
            continue;
        }
        if (static_cast<std::size_t>(token) >= vocab.size()) {
            throw std::runtime_error("Packed n-gram token is outside the vocabulary.");
        }
        std::uint64_t symbol = vocab[token];
        for (int offset = 0; offset < n; ++offset) {
            output.push_back(static_cast<std::uint8_t>((symbol >> (8 * offset)) & 0xFFU));
        }
    }
    if (escape_offset != static_cast<std::size_t>(escape_len)) {
        throw std::runtime_error("Packed n-gram escape stream contains unused bytes.");
    }
    output.insert(output.end(), tail, tail + tail_len);
    return output;
}

std::uint64_t pack_ngram_symbol(const std::uint8_t* data, int n) {
    std::uint64_t symbol = 0;
    for (int index = 0; index < n; ++index) {
        symbol |= static_cast<std::uint64_t>(data[index]) << (8 * index);
    }
    return symbol;
}

std::vector<std::uint64_t> vocab_from_bytes(
    const std::uint8_t* data,
    std::size_t size,
    int n) {
    if (n < 1 || n > 8) {
        throw std::runtime_error("ngram vocab requires n in [1, 8].");
    }
    const Py_ssize_t symbol_bytes = n <= 4 ? 4 : 8;
    if (size % static_cast<std::size_t>(symbol_bytes) != 0) {
        throw std::runtime_error(
            "ngram vocab must contain fixed-width little-endian integer values.");
    }
    std::vector<std::uint64_t> vocab(
        size / static_cast<std::size_t>(symbol_bytes));
    for (std::size_t index = 0; index < vocab.size(); ++index) {
        std::uint64_t value = 0;
        for (Py_ssize_t offset = 0; offset < symbol_bytes; ++offset) {
            value |= static_cast<std::uint64_t>(
                         data[index * static_cast<std::size_t>(symbol_bytes)
                              + static_cast<std::size_t>(offset)])
                     << (8 * offset);
        }
        vocab[index] = value;
    }
    if (vocab.size() > 255) {
        throw std::runtime_error("ngram vocab must have at most 255 entries.");
    }
    if (!std::is_sorted(vocab.begin(), vocab.end())) {
        throw std::runtime_error("ngram vocab must be sorted.");
    }
    return vocab;
}

std::size_t hash_ngram_symbol(std::uint64_t value) {
    // SplitMix64's finalizer keeps small little-endian n-grams from clustering
    // in adjacent slots when the table size is a power of two.
    value ^= value >> 30U;
    value *= 0xBF58476D1CE4E5B9ULL;
    value ^= value >> 27U;
    value *= 0x94D049BB133111EBULL;
    value ^= value >> 31U;
    return static_cast<std::size_t>(value);
}

NGramVocabLookup build_ngram_vocab_lookup(
    const std::vector<std::uint64_t>& vocab) {
    NGramVocabLookup lookup;
    std::size_t required = std::max<std::size_t>(8, vocab.size() * 2);
    std::size_t capacity = 1;
    while (capacity < required) {
        capacity <<= 1U;
    }
    lookup.keys.resize(capacity);
    lookup.tokens.assign(capacity, 255);
    lookup.mask = capacity - 1;

    for (std::size_t token = 0; token < vocab.size(); ++token) {
        std::uint64_t symbol = vocab[token];
        std::size_t slot = hash_ngram_symbol(symbol) & lookup.mask;
        while (lookup.tokens[slot] != 255) {
            // `lower_bound` returned the first id for duplicate vocabulary
            // values. Retain that exact behavior even though normal vocabs
            // are unique.
            if (lookup.keys[slot] == symbol) {
                break;
            }
            slot = (slot + 1) & lookup.mask;
        }
        if (lookup.tokens[slot] == 255) {
            lookup.keys[slot] = symbol;
            lookup.tokens[slot] = static_cast<std::uint8_t>(token);
        }
    }
    return lookup;
}

std::uint8_t find_ngram_token(
    const NGramVocabLookup& lookup,
    std::uint64_t symbol) {
    if (lookup.tokens.empty()) {
        return 255;
    }
    std::size_t slot = hash_ngram_symbol(symbol) & lookup.mask;
    for (std::size_t probe = 0; probe < lookup.tokens.size(); ++probe) {
        std::uint8_t token = lookup.tokens[slot];
        if (token == 255) {
            return 255;
        }
        if (lookup.keys[slot] == symbol) {
            return token;
        }
        slot = (slot + 1) & lookup.mask;
    }
    return 255;
}

struct NGramTokenization {
    std::vector<std::uint8_t> tokens;
    std::vector<std::uint8_t> escapes;
    std::vector<std::uint8_t> tail;
};

NGramTokenization tokenize_ngrams(
    const std::uint8_t* data,
    Py_ssize_t data_len,
    int n,
    const std::vector<std::uint64_t>& vocab,
    const NGramVocabLookup& lookup) {
    if (n < 1 || n > 8) {
        throw std::runtime_error("Huff0 n-gram tokenization supports n in [1, 8].");
    }
    if (vocab.empty()) {
        throw std::runtime_error("ngram vocab is empty.");
    }
    if (data_len < 0) {
        throw std::runtime_error("data length must be non-negative.");
    }

    Py_ssize_t complete_len = (data_len / n) * n;
    Py_ssize_t token_count = complete_len / n;
    NGramTokenization result;
    result.tokens.reserve(static_cast<std::size_t>(token_count));
    result.escapes.reserve(static_cast<std::size_t>(complete_len));
    if (data_len > complete_len) {
        result.tail.assign(data + complete_len, data + data_len);
    }

    for (Py_ssize_t token_index = 0; token_index < token_count; ++token_index) {
        const std::uint8_t* pointer = data + token_index * n;
        std::uint64_t symbol = pack_ngram_symbol(pointer, n);
        const std::uint8_t token = find_ngram_token(lookup, symbol);
        if (token != 255) {
            result.tokens.push_back(token);
            continue;
        }

        result.tokens.push_back(255);
        result.escapes.insert(result.escapes.end(), pointer, pointer + n);
    }
    return result;
}

NGramTokenization tokenize_byte_group_stream_ngrams(
    const std::uint8_t* raw,
    int element_size,
    bool bit_reorder,
    int stream,
    std::size_t start,
    Py_ssize_t data_len,
    int n,
    const NGramVocabLookup& lookup) {
    if (n < 1 || n > 8) {
        throw std::runtime_error("Huff0 n-gram tokenization supports n in [1, 8].");
    }
    if (lookup.tokens.empty()) {
        throw std::runtime_error("ngram vocabulary lookup is empty.");
    }
    if (data_len < 0) {
        throw std::runtime_error("data length must be non-negative.");
    }

    Py_ssize_t complete_len = (data_len / n) * n;
    Py_ssize_t token_count = complete_len / n;
    NGramTokenization result;
    result.tokens.reserve(static_cast<std::size_t>(token_count));
    result.escapes.reserve(static_cast<std::size_t>(complete_len));

    for (Py_ssize_t token_index = 0; token_index < token_count; ++token_index) {
        std::array<std::uint8_t, 8> values{};
        std::uint64_t symbol = 0;
        std::size_t position = start
            + static_cast<std::size_t>(token_index * n);
        for (int offset = 0; offset < n; ++offset) {
            std::uint8_t value = byte_group_stream_byte(
                raw,
                element_size,
                bit_reorder,
                stream,
                position + static_cast<std::size_t>(offset));
            values[static_cast<std::size_t>(offset)] = value;
            symbol |= static_cast<std::uint64_t>(value) << (8 * offset);
        }

        std::uint8_t token = find_ngram_token(lookup, symbol);
        if (token != 255) {
            result.tokens.push_back(token);
            continue;
        }

        result.tokens.push_back(255);
        result.escapes.insert(
            result.escapes.end(), values.begin(), values.begin() + n);
    }

    for (Py_ssize_t offset = complete_len; offset < data_len; ++offset) {
        result.tail.push_back(byte_group_stream_byte(
            raw,
            element_size,
            bit_reorder,
            stream,
            start + static_cast<std::size_t>(offset)));
    }
    return result;
}

NGramBlockResult encode_tokenized_ngram_block(
    NGramTokenization tokens,
    const Huff0Codebook& codebook,
    Py_ssize_t min_saving_bytes,
    std::size_t max_output_bytes) {
    NGramBlockResult result;
    if (tokens.tokens.empty()) {
        return result;
    }

    result.payload = encode_huff0_block(
        tokens.tokens.data(),
        static_cast<Py_ssize_t>(tokens.tokens.size()),
        codebook,
        max_output_bytes);
    if (result.payload.empty()
        || result.payload.size() + static_cast<std::size_t>(min_saving_bytes)
            >= tokens.tokens.size()) {
        result.payload.clear();
        return result;
    }

    result.symbol_count = static_cast<Py_ssize_t>(tokens.tokens.size());
    result.escapes = std::move(tokens.escapes);
    result.tail = std::move(tokens.tail);
    result.encoded = true;
    return result;
}

NGramBlockResult encode_ngram_block(
    const std::uint8_t* data,
    Py_ssize_t data_len,
    int n,
    const std::vector<std::uint64_t>& vocab,
    const Huff0Codebook& codebook,
    Py_ssize_t min_saving_bytes,
    std::size_t max_output_bytes,
    const NGramVocabLookup& lookup) {
    return encode_tokenized_ngram_block(
        tokenize_ngrams(data, data_len, n, vocab, lookup),
        codebook,
        min_saving_bytes,
        max_output_bytes);
}

const char* mixed_mode_name(MixedMode mode) {
    switch (mode) {
        case MixedMode::Raw:
            return "raw";
        case MixedMode::SharedHuff0:
            return "shared_huff0";
        case MixedMode::PackedHuff0:
            return "packed_huff0";
    }
    return "raw";
}

// Keep this in sync with payloads.py::PACKED_CHUNK_METADATA_BYTES.
constexpr Py_ssize_t kPackedMetadataBytes = 41;

std::size_t packed_output_budget(
    std::size_t raw_size,
    double compression_threshold) {
    return static_cast<std::size_t>(std::max<Py_ssize_t>(
        0,
        static_cast<Py_ssize_t>(
            static_cast<double>(raw_size) * compression_threshold)
            - kPackedMetadataBytes));
}

bool finish_packed_block(
    MixedBlockResult& result,
    NGramBlockResult& packed,
    std::size_t raw_size,
    int ngram_n,
    double compression_threshold) {
    const double compressed_bytes = static_cast<double>(kPackedMetadataBytes)
        + static_cast<double>(packed.payload.size())
        + static_cast<double>(packed.escapes.size())
        + static_cast<double>(packed.tail.size());
    if (!packed.encoded
        || compressed_bytes
            >= static_cast<double>(raw_size) * compression_threshold) {
        return false;
    }
    result.mode = MixedMode::PackedHuff0;
    result.payload = std::move(packed.payload);
    result.escapes = std::move(packed.escapes);
    result.tail = std::move(packed.tail);
    result.token_count = packed.symbol_count;
    result.ngram_n = ngram_n;
    return true;
}

void process_mixed_block(
    MixedBlockResult& result,
    const MixedStreamSpec& spec,
    Py_ssize_t min_saving_bytes,
    double compression_threshold) {
    result.mode = MixedMode::Raw;
    result.ngram_n = 1;
    if (spec.mode == MixedMode::Raw) {
        return;
    }
    if (spec.codebook == nullptr) {
        throw std::runtime_error("Mixed shared/packed mode is missing a codebook.");
    }
    if (spec.mode == MixedMode::SharedHuff0) {
        constexpr Py_ssize_t kSharedMetadataBytes = 1 + 8 + 8 + 8;
        Py_ssize_t budget = static_cast<Py_ssize_t>(
            static_cast<double>(result.raw.size()) * compression_threshold)
            - kSharedMetadataBytes;
        if (budget > 0) {
            std::vector<std::uint8_t> payload = encode_huff0_block(
                result.raw.data(),
                static_cast<Py_ssize_t>(result.raw.size()),
                *spec.codebook,
                static_cast<std::size_t>(budget));
            if (
                !payload.empty()
                && payload.size() + static_cast<std::size_t>(min_saving_bytes)
                    < result.raw.size()
                && static_cast<double>(kSharedMetadataBytes)
                        + static_cast<double>(payload.size())
                    < static_cast<double>(result.raw.size())
                        * compression_threshold) {
                result.mode = MixedMode::SharedHuff0;
                result.payload = std::move(payload);
                release_vector(result.raw);
                return;
            }
        }
        // A shared post-setup mode must never build a target-local tree.
        // result already owns the exact raw stream and starts in Raw mode.
        return;
    }

    NGramBlockResult packed = encode_ngram_block(
        result.raw.data(),
        static_cast<Py_ssize_t>(result.raw.size()),
        spec.ngram_n,
        spec.vocab,
        *spec.codebook,
        min_saving_bytes,
        packed_output_budget(result.raw.size(), compression_threshold),
        spec.vocab_lookup);
    if (finish_packed_block(
            result,
            packed,
            result.raw.size(),
            spec.ngram_n,
            compression_threshold)) {
        release_vector(result.raw);
        return;
    }
    // A packed post-setup mode must never build a target-local tree.
    // result already owns the exact raw stream and starts in Raw mode.
}

void process_mixed_packed_block(
    MixedBlockResult& result,
    const MixedStreamSpec& spec,
    const std::uint8_t* raw,
    int element_size,
    bool bit_reorder,
    int stream,
    std::size_t start,
    std::size_t length,
    Py_ssize_t min_saving_bytes,
    double compression_threshold) {
    result.mode = MixedMode::Raw;
    result.ngram_n = 1;
    if (spec.codebook == nullptr) {
        throw std::runtime_error("Mixed packed mode is missing a codebook.");
    }

    NGramBlockResult packed = encode_tokenized_ngram_block(
        tokenize_byte_group_stream_ngrams(
            raw,
            element_size,
            bit_reorder,
            stream,
            start,
            static_cast<Py_ssize_t>(length),
            spec.ngram_n,
            spec.vocab_lookup),
        *spec.codebook,
        min_saving_bytes,
        packed_output_budget(length, compression_threshold));
    if (finish_packed_block(
            result,
            packed,
            length,
            spec.ngram_n,
            compression_threshold)) {
        return;
    }

    // The successful packed path never owns a contiguous reordered stream.
    // Materialize it only when direct raw fallback needs the exact bytes.
    extract_byte_group_stream_block(
        result.raw,
        raw,
        element_size,
        bit_reorder,
        stream,
        start,
        length);
}

void process_mixed_stream_block(
    MixedBlockResult& result,
    const MixedStreamSpec& spec,
    const std::uint8_t* raw,
    int element_size,
    bool bit_reorder,
    int stream,
    std::size_t start,
    std::size_t length,
    Py_ssize_t min_saving_bytes,
    double compression_threshold) {
    result.symbol_count = static_cast<Py_ssize_t>(length);
    if (spec.mode == MixedMode::PackedHuff0) {
        process_mixed_packed_block(
            result,
            spec,
            raw,
            element_size,
            bit_reorder,
            stream,
            start,
            length,
            min_saving_bytes,
            compression_threshold);
        return;
    }

    extract_byte_group_stream_block(
        result.raw,
        raw,
        element_size,
        bit_reorder,
        stream,
        start,
        length);
    process_mixed_block(
        result, spec, min_saving_bytes, compression_threshold);
}

void delete_codebook(void* pointer) noexcept {
    delete static_cast<Huff0Codebook*>(pointer);
}

void delete_decodebook(void* pointer) noexcept {
    delete static_cast<Huff0Decodebook*>(pointer);
}

bool truthy(nb::handle source) {
    return static_cast<bool>(nb::bool_(source));
}

Huff0Codebook* codebook_from_capsule(nb::handle source) {
    if (!nb::isinstance<nb::capsule>(source)) {
        throw nb::value_error("Invalid Huff0 codebook capsule.");
    }
    nb::capsule capsule = nb::borrow<nb::capsule>(source);
    return static_cast<Huff0Codebook*>(
        capsule.data("neuralzip.Huff0Codebook"));
}

Huff0Decodebook* decodebook_from_capsule(nb::handle source) {
    if (!nb::isinstance<nb::capsule>(source)) {
        throw nb::value_error("Invalid Huff0 decodebook capsule.");
    }
    nb::capsule capsule = nb::borrow<nb::capsule>(source);
    return static_cast<Huff0Decodebook*>(
        capsule.data("neuralzip.Huff0Decodebook"));
}

template <typename Function>
decltype(auto) with_value_errors(Function&& function) {
    try {
        return function();
    } catch (const std::bad_alloc&) {
        throw;
    } catch (const std::overflow_error&) {
        throw;
    } catch (const nb::python_error&) {
        throw;
    } catch (const nb::builtin_exception&) {
        throw;
    } catch (const std::exception& error) {
        throw nb::value_error(error.what());
    }
}

std::vector<MixedStreamSpec> parse_mixed_stream_specs(
    nb::handle mode_source,
    int element_size,
    std::size_t base_chunk_size) {
    nb::tuple mode_sequence(mode_source);
    if (mode_sequence.size() != static_cast<std::size_t>(element_size)) {
        throw nb::value_error(
            "stream_modes must contain exactly one mode per byte stream.");
    }

    std::vector<MixedStreamSpec> specs(
        static_cast<std::size_t>(element_size));
    for (int stream = 0; stream < element_size; ++stream) {
        nb::tuple item(nb::handle(mode_sequence[stream].ptr()));
        if (item.size() != 4) {
            throw nb::type_error(
                "Each stream mode must contain kind, codebook, vocabulary, "
                "and ngram_n.");
        }

        std::string kind = nb::cast<std::string>(
            nb::handle(item[0].ptr()));
        nb::handle codebook_source(item[1].ptr());
        nb::handle vocab_source(item[2].ptr());
        const int ngram_n = nb::cast<int>(nb::handle(item[3].ptr()));

        MixedStreamSpec& spec = specs[static_cast<std::size_t>(stream)];
        spec.ngram_n = ngram_n;
        if (kind == "raw") {
            spec.mode = MixedMode::Raw;
            spec.ngram_n = 1;
        } else if (kind == "shared_huff0") {
            spec.mode = MixedMode::SharedHuff0;
            spec.codebook_owner = nb::borrow<nb::object>(codebook_source);
            spec.codebook = codebook_from_capsule(codebook_source);
            spec.ngram_n = 1;
        } else if (kind == "packed_huff0") {
            if (ngram_n < 2 || ngram_n > 8) {
                throw nb::value_error(
                    "packed_huff0 requires ngram_n in [2, 8].");
            }
            spec.mode = MixedMode::PackedHuff0;
            spec.codebook_owner = nb::borrow<nb::object>(codebook_source);
            spec.codebook = codebook_from_capsule(codebook_source);
            ReadOnlyBuffer vocab(vocab_source);
            spec.vocab = with_value_errors([&]() {
                return vocab_from_bytes(
                    vocab.data(),
                    vocab.size(),
                    ngram_n);
            });
            if (spec.vocab.empty()) {
                throw nb::value_error(
                    "packed_huff0 vocabulary is empty.");
            }
            spec.vocab_lookup = build_ngram_vocab_lookup(spec.vocab);
        } else {
            throw nb::value_error("Unsupported mixed Huff0 stream mode.");
        }

        if (spec.mode == MixedMode::Raw) {
            spec.chunk_size = base_chunk_size;
        } else if (spec.mode == MixedMode::SharedHuff0) {
            spec.chunk_size = std::min(base_chunk_size, kMaxBlockSize);
        } else {
            const std::size_t n = static_cast<std::size_t>(spec.ngram_n);
            const std::size_t limit = std::min(
                base_chunk_size, kMaxBlockSize * n);
            spec.chunk_size = std::max(n, (limit / n) * n);
        }
    }
    return specs;
}

nb::list mixed_groups_to_python(
    int element_size,
    const std::vector<std::size_t>& group_sizes,
    const std::vector<MixedBlockResult>& results,
    std::size_t result_offset) {
    nb::list groups;
    std::size_t result_index = result_offset;
    for (int stream = 0; stream < element_size; ++stream) {
        nb::list chunks;
        const std::size_t group_size =
            group_sizes[static_cast<std::size_t>(stream)];
        for (std::size_t index = 0; index < group_size;
             ++index, ++result_index) {
            if (result_index >= results.size()) {
                throw std::runtime_error(
                    "Mixed-stream result index exceeds the native result set.");
            }
            const MixedBlockResult& result = results[result_index];
            const auto& body = result.mode == MixedMode::Raw
                ? result.raw
                : result.payload;
            nb::object token_count = result.mode == MixedMode::PackedHuff0
                ? nb::cast(result.token_count)
                : nb::none();
            chunks.append(nb::make_tuple(
                mixed_mode_name(result.mode),
                bytes_from_vector(body),
                result.symbol_count,
                std::move(token_count),
                bytes_from_vector(result.escapes),
                bytes_from_vector(result.tail),
                result.ngram_n));
        }
        groups.append(std::move(chunks));
    }
    return groups;
}

nb::tuple codebook_to_python(std::unique_ptr<Huff0Codebook> codebook) {
    Huff0Codebook* native = codebook.get();
    nb::capsule handle(
        native, "neuralzip.Huff0Codebook", delete_codebook);
    codebook.release();
    return nb::make_tuple(
        std::move(handle), bytes_from_vector(native->serialized));
}

void validate_codebook_options(
    Py_ssize_t prior_count,
    Py_ssize_t max_bits) {
    if (prior_count < 0 || prior_count > static_cast<Py_ssize_t>(
            std::numeric_limits<unsigned>::max())) {
        throw nb::value_error("prior_count must fit in unsigned.");
    }
    if (max_bits < 1
        || max_bits > static_cast<Py_ssize_t>(kMaxTableLog)) {
        throw nb::value_error("max_bits must be in [1, 12].");
    }
}

std::array<unsigned, 256> parse_symbol_counts(
    nb::handle source,
    unsigned prior_count) {
    nb::tuple sequence(source);
    if (sequence.size() != 256) {
        throw nb::value_error(
            "symbol_counts must contain exactly 256 entries.");
    }

    std::array<std::uint64_t, 256> counts{};
    std::uint64_t total = 0;
    for (std::size_t index = 0; index < counts.size(); ++index) {
        const auto count = PyLong_AsUnsignedLongLong(
            sequence[index].ptr());
        if (PyErr_Occurred()) {
            throw nb::python_error();
        }
        if (total > std::numeric_limits<std::uint64_t>::max() - count) {
            throw std::overflow_error("Huff0 symbol count overflow.");
        }
        counts[index] = count;
        total += count;
    }

    const std::uint64_t prior_total =
        static_cast<std::uint64_t>(prior_count) * counts.size();
    if (prior_total >= kMaxHistogramTotal) {
        throw nb::value_error(
            "prior_count is too large for a 256-symbol Huff0 table.");
    }
    const std::uint64_t available = kMaxHistogramTotal - prior_total;

    unsigned shift = 0;
    if (total > available) {
        for (shift = 1; shift < 64; ++shift) {
            std::uint64_t scaled_total = 0;
            for (const std::uint64_t count : counts) {
                if (count != 0) {
                    scaled_total += std::max<std::uint64_t>(
                        1, count >> shift);
                }
            }
            if (scaled_total <= available) {
                break;
            }
        }
        if (shift == 64) {
            throw std::overflow_error("Huff0 symbol count overflow.");
        }
    }

    std::array<unsigned, 256> output{};
    for (std::size_t index = 0; index < counts.size(); ++index) {
        const std::uint64_t count = counts[index];
        const std::uint64_t scaled =
            count == 0 ? 0 : std::max<std::uint64_t>(1, count >> shift);
        output[index] = static_cast<unsigned>(scaled + prior_count);
    }
    return output;
}

std::size_t max_block_size() {
    return kMaxBlockSize;
}

nb::tuple build_codebook(
    nb::handle data_source,
    Py_ssize_t prior_count,
    Py_ssize_t max_bits) {
    validate_codebook_options(prior_count, max_bits);
    ReadOnlyBuffer data(data_source);
    return with_value_errors([&]() {
        return codebook_to_python(make_codebook(
            data.data(),
            data.size(),
            static_cast<unsigned>(prior_count),
            static_cast<unsigned>(max_bits)));
    });
}

nb::tuple build_codebook_from_counts(
    nb::handle count_source,
    Py_ssize_t prior_count,
    Py_ssize_t max_bits) {
    validate_codebook_options(prior_count, max_bits);
    std::array<unsigned, 256> counts = parse_symbol_counts(
        count_source, static_cast<unsigned>(prior_count));
    return with_value_errors([&]() {
        return codebook_to_python(make_codebook_from_counts(
            counts, static_cast<unsigned>(max_bits)));
    });
}

nb::tuple build_codebook_from_header(nb::handle header_source) {
    ReadOnlyBuffer header(header_source);
    return with_value_errors([&]() {
        return codebook_to_python(make_codebook_from_header(
            header.data(),
            header.size()));
    });
}

nb::capsule build_decodebook(nb::handle header_source) {
    ReadOnlyBuffer header(header_source);
    return with_value_errors([&]() {
        auto decodebook = make_decodebook_from_header(
            header.data(),
            header.size());
        Huff0Decodebook* native = decodebook.get();
        nb::capsule handle(
            native, "neuralzip.Huff0Decodebook", delete_decodebook);
        decodebook.release();
        return handle;
    });
}

nb::tuple export_code_table(
    nb::handle header_source,
    Py_ssize_t max_bits) {
    if (max_bits <= 0 || max_bits > 12) {
        throw nb::value_error("max_bits must be in [1, 12].");
    }
    ReadOnlyBuffer header(header_source);
    return with_value_errors([&]() {
        const ExportedCodeTable table = export_code_table_from_header(
            header.data(),
            header.size(),
            static_cast<unsigned>(max_bits));
        nb::list values;
        for (const auto value : table.code_values) {
            values.append(value);
        }
        return nb::make_tuple(
            nb::bytes(table.code_lengths.data(), table.code_lengths.size()),
            nb::tuple(values),
            table.max_bits);
    });
}

nb::tuple encode_gpu_bfloat16(
    nb::handle raw_source,
    nb::handle header_source,
    Py_ssize_t bytes_per_thread,
    Py_ssize_t threads_per_block) {
    if (bytes_per_thread <= 0 || threads_per_block <= 0) {
        throw nb::value_error(
            "bytes_per_thread and threads_per_block must be positive.");
    }
    if (bytes_per_thread != 8) {
        throw nb::value_error(
            "bytes_per_thread must be 8 for this ABI.");
    }
    if (threads_per_block != 512) {
        throw nb::value_error(
            "threads_per_block must be 512 for this ABI.");
    }
    if (static_cast<std::uint64_t>(bytes_per_thread)
            > std::numeric_limits<std::uint32_t>::max()
        || static_cast<std::uint64_t>(threads_per_block)
            > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error(
            "GPU layout dimensions must fit in uint32.");
    }

    ReadOnlyBuffer raw(raw_source);
    ReadOnlyBuffer header(header_source);
    if ((raw.size() & 1) != 0) {
        throw nb::value_error(
            "A BF16 byte sequence must have an even length.");
    }

    const std::uint8_t* raw_data = raw.data();
    const std::uint8_t* header_data = header.data();
    std::vector<std::uint8_t> raw_snapshot;
    std::vector<std::uint8_t> header_snapshot;
    if (!raw.readonly() && raw.size() > 0) {
        raw_snapshot.assign(raw.data(), raw.data() + raw.size());
        raw_data = raw_snapshot.data();
    }
    if (!header.readonly() && header.size() > 0) {
        header_snapshot.assign(
            header.data(), header.data() + header.size());
        header_data = header_snapshot.data();
    }

    GPUBFloat16Encoding encoded = with_value_errors([&]() {
        nb::gil_scoped_release release;
        return encode_gpu_bfloat16_layout(
            raw_data,
            static_cast<std::size_t>(raw.size()),
            header_data,
            header.size(),
            static_cast<std::uint32_t>(bytes_per_thread),
            static_cast<std::uint32_t>(threads_per_block));
    });

    nb::list positions;
    for (const auto position : encoded.output_positions) {
        positions.append(position);
    }
    return nb::make_tuple(
        bytes_from_vector(encoded.encoded_exponents),
        bytes_from_vector(encoded.sign_mantissa),
        bytes_from_vector(encoded.packed_gaps),
        nb::tuple(positions),
        encoded.bit_length,
        encoded.max_bits);
}

nb::object encode_with_codebook(
    nb::handle data_source,
    nb::capsule codebook_handle,
    Py_ssize_t min_saving_bytes,
    Py_ssize_t max_output_bytes) {
    if (min_saving_bytes < 0) {
        throw nb::value_error(
            "min_saving_bytes must be non-negative.");
    }
    ReadOnlyBuffer data(data_source);
    Huff0Codebook* codebook = codebook_from_capsule(codebook_handle);
    std::vector<std::uint8_t> encoded = with_value_errors([&]() {
        const std::size_t max_bytes = max_output_bytes < 0
            ? std::numeric_limits<std::size_t>::max()
            : static_cast<std::size_t>(max_output_bytes);
        return encode_huff0_block(
            data.data(),
            data.size(),
            *codebook,
            max_bytes);
    });
    if (encoded.empty()
        || encoded.size() + static_cast<std::size_t>(min_saving_bytes)
            >= data.size()) {
        return nb::none();
    }
    return nb::make_tuple(
        bytes_from_vector(encoded),
        static_cast<Py_ssize_t>(data.size()));
}

nb::object encode_ngram_with_codebook(
    nb::handle data_source,
    nb::capsule codebook_handle,
    nb::handle vocab_source,
    Py_ssize_t n,
    Py_ssize_t min_saving_bytes,
    Py_ssize_t max_output_bytes) {
    if (n < 1 || n > 8) {
        throw nb::value_error("n must be in [1, 8].");
    }
    if (min_saving_bytes < 0) {
        throw nb::value_error(
            "min_saving_bytes must be non-negative.");
    }
    ReadOnlyBuffer data(data_source);
    ReadOnlyBuffer vocab_bytes(vocab_source);
    Huff0Codebook* codebook = codebook_from_capsule(codebook_handle);
    NGramBlockResult encoded = with_value_errors([&]() {
        std::vector<std::uint64_t> vocab = vocab_from_bytes(
            vocab_bytes.data(),
            vocab_bytes.size(),
            static_cast<int>(n));
        NGramVocabLookup vocab_lookup = build_ngram_vocab_lookup(vocab);
        const std::size_t max_bytes = max_output_bytes < 0
            ? std::numeric_limits<std::size_t>::max()
            : static_cast<std::size_t>(max_output_bytes);
        return encode_ngram_block(
            data.data(),
            data.size(),
            static_cast<int>(n),
            vocab,
            *codebook,
            min_saving_bytes,
            max_bytes,
            vocab_lookup);
    });
    if (!encoded.encoded) {
        return nb::none();
    }
    return ngram_result_to_tuple(encoded);
}

nb::bytes decompress_with_decodebook(
    nb::handle payload_source,
    nb::capsule decodebook_handle,
    Py_ssize_t symbol_count) {
    ReadOnlyBuffer payload(payload_source);
    Huff0Decodebook* decodebook =
        decodebook_from_capsule(decodebook_handle);
    return with_value_errors([&]() {
        return bytes_from_vector(decode_huff0_block(
            payload.data(),
            payload.size(),
            symbol_count,
            *decodebook));
    });
}

nb::bytes decompress_ngram_with_decodebook(
    nb::handle payload_source,
    nb::capsule decodebook_handle,
    Py_ssize_t symbol_count,
    nb::handle vocab_source,
    Py_ssize_t n,
    nb::handle escape_source,
    nb::handle tail_source) {
    ReadOnlyBuffer payload(payload_source);
    ReadOnlyBuffer vocab_bytes(vocab_source);
    ReadOnlyBuffer escapes(escape_source);
    ReadOnlyBuffer tail(tail_source);
    Huff0Decodebook* decodebook =
        decodebook_from_capsule(decodebook_handle);
    return with_value_errors([&]() {
        std::vector<std::uint8_t> tokens = decode_huff0_block(
            payload.data(),
            payload.size(),
            symbol_count,
            *decodebook);
        std::vector<std::uint64_t> vocab = vocab_from_bytes(
            vocab_bytes.data(),
            vocab_bytes.size(),
            static_cast<int>(n));
        return bytes_from_vector(restore_ngram_tokens(
            tokens,
            static_cast<int>(n),
            vocab,
            escapes.data(),
            escapes.size(),
            tail.data(),
            tail.size()));
    });
}

nb::list compress_mixed_streams_many(
    nb::handle tensor_source,
    Py_ssize_t min_saving_bytes,
    int requested_threads,
    double compression_threshold) {
    if (min_saving_bytes < 0 || requested_threads < 0) {
        throw nb::value_error(
            "min_saving_bytes and threads must be non-negative.");
    }
    if (!(compression_threshold > 0.0
          && compression_threshold <= 1.0)) {
        throw nb::value_error(
            "compression_threshold must be in (0, 1].");
    }

    nb::tuple tensor_sequence(tensor_source);
    struct MixedTensorRequest {
        ReadOnlyBuffer raw;
        int element_size = 0;
        bool bit_reorder = false;
        std::vector<MixedStreamSpec> specs;
        std::vector<std::size_t> group_sizes;
        std::size_t result_offset = 0;
    };
    struct MixedModelJob {
        std::size_t tensor_index;
        int stream;
        std::size_t start;
        std::size_t length;
    };

    std::vector<MixedTensorRequest> requests(tensor_sequence.size());
    std::vector<MixedModelJob> jobs;
    std::size_t total_input_bytes = 0;
    for (std::size_t tensor_index = 0;
         tensor_index < tensor_sequence.size();
         ++tensor_index) {
        nb::tuple item(nb::handle(tensor_sequence[tensor_index].ptr()));
        if (item.size() != 5) {
            throw nb::type_error(
                "Each tensor spec must contain data, element_size, "
                "bit_reorder, chunk_size, and stream_modes.");
        }
        nb::handle raw_source(item[0].ptr());
        const int element_size =
            nb::cast<int>(nb::handle(item[1].ptr()));
        const bool bit_reorder = truthy(nb::handle(item[2].ptr()));
        const Py_ssize_t requested_chunk_size =
            nb::cast<Py_ssize_t>(nb::handle(item[3].ptr()));
        nb::handle mode_source(item[4].ptr());

        if (element_size != 2 && element_size != 4) {
            throw nb::value_error("element_size must be 2 or 4.");
        }
        if (requested_chunk_size <= 0) {
            throw nb::value_error("chunk_size must be positive.");
        }

        MixedTensorRequest& request = requests[tensor_index];
        request.raw.acquire(raw_source);
        if (request.raw.size() % element_size != 0) {
            throw nb::value_error(
                "data length must be divisible by element_size.");
        }
        const std::size_t raw_bytes =
            static_cast<std::size_t>(request.raw.size());
        if (raw_bytes
            > std::numeric_limits<std::size_t>::max()
                - total_input_bytes) {
            throw std::overflow_error(
                "Total mixed input size overflowed.");
        }
        total_input_bytes += raw_bytes;
        request.element_size = element_size;
        request.bit_reorder = bit_reorder;
        request.specs = parse_mixed_stream_specs(
            mode_source,
            element_size,
            static_cast<std::size_t>(requested_chunk_size));
        request.group_sizes.assign(
            static_cast<std::size_t>(element_size), 0);
        request.result_offset = jobs.size();

        const std::size_t symbol_count =
            raw_bytes / static_cast<std::size_t>(element_size);
        for (int stream = 0; stream < element_size; ++stream) {
            const std::size_t step =
                request.specs[static_cast<std::size_t>(stream)].chunk_size;
            for (std::size_t start = 0;
                 start < symbol_count;
                 start += step) {
                jobs.push_back({
                    tensor_index,
                    stream,
                    start,
                    std::min(step, symbol_count - start),
                });
                request.group_sizes[
                    static_cast<std::size_t>(stream)] += 1;
            }
        }
    }

    std::vector<MixedBlockResult> results(jobs.size());
    std::string thread_error;
    {
        nb::gil_scoped_release release;
        thread_error = run_parallel_jobs(
            jobs.size(),
            total_input_bytes,
            requested_threads,
            [&](std::size_t job_index) {
                const MixedModelJob& job = jobs[job_index];
                const MixedTensorRequest& request =
                    requests[job.tensor_index];
                const MixedStreamSpec& spec = request.specs[
                    static_cast<std::size_t>(job.stream)];
                MixedBlockResult& result = results[job_index];
                try {
                    process_mixed_stream_block(
                        result,
                        spec,
                        request.raw.data(),
                        request.element_size,
                        request.bit_reorder,
                        job.stream,
                        job.start,
                        job.length,
                        min_saving_bytes,
                        compression_threshold);
                } catch (const std::exception& error) {
                    result.error = error.what();
                } catch (...) {
                    result.error =
                        "Unknown model-wide mixed encoding failure.";
                }
            });
    }

    if (!thread_error.empty()) {
        throw std::runtime_error(thread_error);
    }
    for (const auto& result : results) {
        if (!result.error.empty()) {
            throw nb::value_error(result.error.c_str());
        }
    }

    nb::list output;
    for (const auto& request : requests) {
        output.append(mixed_groups_to_python(
            request.element_size,
            request.group_sizes,
            results,
            request.result_offset));
    }
    return output;
}

nb::list decompress_mixed_streams_many(
    nb::handle tensor_source,
    int requested_threads) {
    if (requested_threads < 0) {
        throw nb::value_error("threads must be non-negative.");
    }

    nb::tuple tensor_sequence(tensor_source);
    const std::size_t tensor_count = tensor_sequence.size();
    std::vector<MixedStreamDecodeTensor> requests(tensor_count);
    std::vector<MixedStreamDecodeChunk> chunks;
    std::vector<MixedStreamAlignedDecodeJob> aligned_jobs;
    std::vector<std::size_t> fallback_chunk_jobs;
    std::vector<MixedStreamCombineJob> combine_jobs;

    std::size_t total_output_bytes = 0;
    for (std::size_t tensor_index = 0;
         tensor_index < tensor_count;
         ++tensor_index) {
        nb::tuple item(nb::handle(tensor_sequence[tensor_index].ptr()));
        if (item.size() != 4) {
            throw nb::type_error(
                "Each decode spec must contain element_size, bit_reorder, "
                "numel, and groups.");
        }
        const int element_size =
            nb::cast<int>(nb::handle(item[0].ptr()));
        const bool bit_reorder = truthy(nb::handle(item[1].ptr()));
        const Py_ssize_t requested_numel =
            nb::cast<Py_ssize_t>(nb::handle(item[2].ptr()));
        nb::handle groups_source(item[3].ptr());

        if (element_size != 2 && element_size != 4) {
            throw nb::value_error("element_size must be 2 or 4.");
        }
        if (requested_numel < 0) {
            throw nb::value_error("numel must be non-negative.");
        }

        const std::size_t numel =
            static_cast<std::size_t>(requested_numel);
        const std::size_t width =
            static_cast<std::size_t>(element_size);
        if (numel > static_cast<std::size_t>(PY_SSIZE_T_MAX) / width) {
            throw std::overflow_error(
                "Decoded tensor byte size exceeds Py_ssize_t.");
        }
        const std::size_t raw_bytes = numel * width;
        if (raw_bytes
            > std::numeric_limits<std::size_t>::max()
                - total_output_bytes) {
            throw std::overflow_error(
                "Total decoded tensor size overflowed.");
        }
        total_output_bytes += raw_bytes;

        MixedStreamDecodeTensor& request = requests[tensor_index];
        request.element_size = element_size;
        request.bit_reorder = bit_reorder;
        request.numel = numel;
        nb::bytearray output(
            static_cast<const void*>(nullptr), raw_bytes);
        request.output_data =
            static_cast<std::uint8_t*>(output.data());
        request.output = std::move(output);

        nb::tuple group_sequence(groups_source);
        if (group_sequence.size()
            != static_cast<std::size_t>(element_size)) {
            throw nb::value_error(
                "groups must contain exactly one group per byte stream.");
        }

        std::array<std::vector<std::size_t>, 4>
            stream_chunk_indices;
        for (int stream = 0; stream < element_size; ++stream) {
            nb::tuple chunk_sequence(
                nb::handle(group_sequence[stream].ptr()));
            std::size_t output_offset = 0;
            for (std::size_t chunk_index = 0;
                 chunk_index < chunk_sequence.size();
                 ++chunk_index) {
                nb::tuple chunk_source(
                    nb::handle(chunk_sequence[chunk_index].ptr()));
                if (chunk_source.size() != 4) {
                    throw nb::type_error(
                        "Each chunk must contain mode, payload, "
                        "symbol_count, and decodebook.");
                }
                const std::string kind = nb::cast<std::string>(
                    nb::handle(chunk_source[0].ptr()));
                nb::handle payload_source(chunk_source[1].ptr());
                const Py_ssize_t requested_symbols =
                    nb::cast<Py_ssize_t>(
                        nb::handle(chunk_source[2].ptr()));
                nb::handle decodebook_source(chunk_source[3].ptr());

                if (requested_symbols < 0) {
                    throw nb::value_error(
                        "chunk symbol_count must be non-negative.");
                }
                const std::size_t symbol_count =
                    static_cast<std::size_t>(requested_symbols);
                if (symbol_count > numel - output_offset) {
                    throw nb::value_error(
                        "chunk symbol counts exceed the tensor numel.");
                }

                chunks.emplace_back();
                MixedStreamDecodeChunk& chunk = chunks.back();
                chunk.tensor_index = tensor_index;
                chunk.stream = stream;
                chunk.output_offset = output_offset;
                chunk.symbol_count = symbol_count;
                chunk.payload.acquire(payload_source);

                if (kind == "raw") {
                    chunk.mode = MixedStreamDecodeMode::Raw;
                    if (!decodebook_source.is_none()) {
                        throw nb::value_error(
                            "raw chunks cannot reference a decodebook.");
                    }
                    if (chunk.payload.size() != requested_symbols) {
                        throw nb::value_error(
                            "raw chunk payload size must equal symbol_count.");
                    }
                } else if (kind == "shared_huff0") {
                    chunk.mode = MixedStreamDecodeMode::SharedHuff0;
                    if (decodebook_source.is_none()) {
                        throw nb::value_error(
                            "shared_huff0 chunks require a "
                            "decodebook handle.");
                    }
                    if (symbol_count == 0 || chunk.payload.size() <= 0) {
                        throw nb::value_error(
                            "shared_huff0 chunks must contain symbols "
                            "and payload.");
                    }
                    if (symbol_count > kMaxBlockSize) {
                        throw nb::value_error(
                            "shared_huff0 chunk exceeds the "
                            "Huff0 block limit.");
                    }
                    chunk.decodebook_owner =
                        nb::borrow<nb::object>(decodebook_source);
                    chunk.decodebook =
                        decodebook_from_capsule(decodebook_source);
                } else {
                    throw nb::value_error(
                        "Unsupported mixed-stream decode chunk mode.");
                }

                stream_chunk_indices[
                    static_cast<std::size_t>(stream)].push_back(
                        chunks.size() - 1U);
                output_offset += symbol_count;
            }
            if (output_offset != numel) {
                throw nb::value_error(
                    "Each decoded byte stream must contain exactly "
                    "numel symbols.");
            }
        }

        bool chunks_are_aligned = true;
        const std::size_t aligned_chunk_count =
            stream_chunk_indices[0].size();
        for (std::size_t stream = 1; stream < width; ++stream) {
            if (stream_chunk_indices[stream].size()
                != aligned_chunk_count) {
                chunks_are_aligned = false;
                break;
            }
        }
        if (chunks_are_aligned) {
            for (std::size_t chunk_index = 0;
                 chunk_index < aligned_chunk_count;
                 ++chunk_index) {
                const MixedStreamDecodeChunk& first = chunks[
                    stream_chunk_indices[0][chunk_index]];
                for (std::size_t stream = 1;
                     stream < width;
                     ++stream) {
                    const MixedStreamDecodeChunk& other = chunks[
                        stream_chunk_indices[stream][chunk_index]];
                    if (other.output_offset != first.output_offset
                        || other.symbol_count != first.symbol_count) {
                        chunks_are_aligned = false;
                        break;
                    }
                }
                if (!chunks_are_aligned) {
                    break;
                }
            }
        }

        if (chunks_are_aligned) {
            for (std::size_t chunk_index = 0;
                 chunk_index < aligned_chunk_count;
                 ++chunk_index) {
                MixedStreamAlignedDecodeJob job;
                const MixedStreamDecodeChunk& first = chunks[
                    stream_chunk_indices[0][chunk_index]];
                job.tensor_index = tensor_index;
                job.output_offset = first.output_offset;
                job.symbol_count = first.symbol_count;
                for (std::size_t stream = 0;
                     stream < width;
                     ++stream) {
                    job.chunk_indices[stream] =
                        stream_chunk_indices[stream][chunk_index];
                }
                aligned_jobs.push_back(std::move(job));
            }
        } else {
            request.lanes.resize(width);
            for (auto& lane : request.lanes) {
                lane.resize(numel);
            }
            for (std::size_t stream = 0;
                 stream < width;
                 ++stream) {
                fallback_chunk_jobs.insert(
                    fallback_chunk_jobs.end(),
                    stream_chunk_indices[stream].begin(),
                    stream_chunk_indices[stream].end());
            }

            constexpr std::size_t kCombineJobBytes = 256U << 10;
            const std::size_t combine_step = std::max<std::size_t>(
                1U, kCombineJobBytes / width);
            for (std::size_t start = 0;
                 start < numel;
                 start += combine_step) {
                combine_jobs.push_back({
                    tensor_index,
                    start,
                    std::min(combine_step, numel - start),
                });
            }
        }
    }

    std::string thread_error;
    std::string native_error;
    const std::string* decode_error = nullptr;
    {
        nb::gil_scoped_release release;
        try {
            const std::size_t aligned_job_count = aligned_jobs.size();
            const std::size_t first_phase_job_count =
                aligned_job_count + fallback_chunk_jobs.size();
            thread_error = run_parallel_jobs(
                first_phase_job_count,
                total_output_bytes,
                requested_threads,
                [&](std::size_t job_index) {
                    if (job_index < aligned_job_count) {
                        MixedStreamAlignedDecodeJob& job =
                            aligned_jobs[job_index];
                        const MixedStreamDecodeTensor& request =
                            requests[job.tensor_index];
                        try {
                            const std::size_t width =
                                static_cast<std::size_t>(
                                    request.element_size);
                            std::size_t huff_lane_count = 0;
                            for (std::size_t stream = 0;
                                 stream < width;
                                 ++stream) {
                                const MixedStreamDecodeMode mode =
                                    chunks[job.chunk_indices[stream]].mode;
                                if (mode == MixedStreamDecodeMode::SharedHuff0) {
                                    ++huff_lane_count;
                                }
                            }
                            if (huff_lane_count > 0
                                && job.symbol_count
                                    > std::numeric_limits<std::size_t>::max()
                                        / huff_lane_count) {
                                throw std::length_error(
                                    "Aligned Huff0 scratch size overflowed.");
                            }

                            thread_local std::vector<std::uint8_t> scratch;
                            scratch.resize(
                                huff_lane_count * job.symbol_count);
                            std::array<const std::uint8_t*, 4> lane_data{};
                            std::size_t scratch_offset = 0;
                            for (std::size_t stream = 0;
                                 stream < width;
                                 ++stream) {
                                MixedStreamDecodeChunk& chunk = chunks[
                                    job.chunk_indices[stream]];
                                const auto* payload = chunk.payload.data();
                                if (chunk.mode == MixedStreamDecodeMode::Raw) {
                                    lane_data[stream] = payload;
                                    continue;
                                }
                                std::uint8_t* destination =
                                    scratch.data() + scratch_offset;
                                decode_huff0_block_into(
                                    payload,
                                    chunk.payload.size(),
                                    destination,
                                    chunk.symbol_count,
                                    *chunk.decodebook);
                                lane_data[stream] = destination;
                                scratch_offset += chunk.symbol_count;
                            }
                            restore_mixed_stream_chunk(
                                request,
                                lane_data,
                                job.output_offset,
                                job.symbol_count);
                        } catch (const std::exception& error) {
                            job.error = error.what();
                        } catch (...) {
                            job.error =
                                "Unknown aligned mixed-stream decode failure.";
                        }
                        return;
                    }

                    MixedStreamDecodeChunk& chunk = chunks[
                        fallback_chunk_jobs[
                            job_index - aligned_job_count]];
                    MixedStreamDecodeTensor& request =
                        requests[chunk.tensor_index];
                    std::uint8_t* destination = nullptr;
                    if (chunk.symbol_count > 0) {
                        destination = request.lanes[
                            static_cast<std::size_t>(chunk.stream)].data()
                            + chunk.output_offset;
                    }
                    try {
                        const auto* payload = chunk.payload.data();
                        if (chunk.mode == MixedStreamDecodeMode::Raw) {
                            if (chunk.symbol_count > 0) {
                                std::memcpy(
                                    destination,
                                    payload,
                                    chunk.symbol_count);
                            }
                        } else {
                            decode_huff0_block_into(
                                payload,
                                chunk.payload.size(),
                                destination,
                                chunk.symbol_count,
                                *chunk.decodebook);
                        }
                    } catch (const std::exception& error) {
                        chunk.error = error.what();
                    } catch (...) {
                        chunk.error =
                            "Unknown mixed-stream decode failure.";
                    }
                });

            if (thread_error.empty()) {
                for (const auto& job : aligned_jobs) {
                    if (!job.error.empty()) {
                        decode_error = &job.error;
                        break;
                    }
                }
            }
            if (thread_error.empty() && decode_error == nullptr) {
                for (const auto& chunk : chunks) {
                    if (!chunk.error.empty()) {
                        decode_error = &chunk.error;
                        break;
                    }
                }
            }
            if (thread_error.empty() && decode_error == nullptr) {
                thread_error = run_parallel_jobs(
                    combine_jobs.size(),
                    total_output_bytes,
                    requested_threads,
                    [&](std::size_t job_index) {
                        const MixedStreamCombineJob& job =
                            combine_jobs[job_index];
                        restore_mixed_stream_tensor_range(
                            requests[job.tensor_index],
                            job.start,
                            job.length);
                    });
            }
        } catch (const std::exception& error) {
            native_error = error.what();
        } catch (...) {
            native_error =
                "Unknown model-wide mixed-stream decode failure.";
        }
    }

    if (!native_error.empty()) {
        throw std::runtime_error(native_error);
    }
    if (!thread_error.empty()) {
        throw std::runtime_error(thread_error);
    }
    if (decode_error != nullptr) {
        throw nb::value_error(decode_error->c_str());
    }

    nb::list output;
    for (const auto& request : requests) {
        output.append(request.output);
    }
    return output;
}

}  // namespace

NB_MODULE(_huff0_cpp, module) {
    module.doc() =
        "NeuralZip stream bridge backed by upstream Zstd Huff0.";

    module.def(
        "max_block_size",
        &max_block_size,
        "Return Huff0 maximum block size.");
    module.def(
        "build_codebook",
        &build_codebook,
        nb::arg("data"),
        nb::arg("prior_count") = 0,
        nb::arg("max_bits") = kTableLog,
        "Build a Huff0 codebook.");
    module.def(
        "build_codebook_from_counts",
        &build_codebook_from_counts,
        nb::arg("symbol_counts"),
        nb::arg("prior_count") = 0,
        nb::arg("max_bits") = kTableLog,
        "Build a Huff0 codebook from a complete 256-bin histogram.");
    module.def(
        "build_codebook_from_header",
        &build_codebook_from_header,
        "Rebuild a Huff0 encoder table from its serialized header.");
    module.def(
        "build_decodebook",
        &build_decodebook,
        "Build a Huff0 decode table from a serialized codebook.");
    module.def(
        "export_code_table",
        &export_code_table,
        nb::arg("codebook"),
        nb::arg("max_bits") = 12,
        "Export exact Huff0 symbol lengths and canonical code values.");
    module.def(
        "encode_gpu_bfloat16",
        &encode_gpu_bfloat16,
        nb::arg("raw_bf16"),
        nb::arg("codebook"),
        nb::arg("bytes_per_thread") = 8,
        nb::arg("threads_per_block") = 512,
        "Encode little-endian BF16 bytes into the NeuralZip GPU layout.");
    module.def(
        "encode_with_codebook",
        &encode_with_codebook,
        "Encode a four-stream frame using a Huff0 codebook.");
    module.def(
        "encode_ngram_with_codebook",
        &encode_ngram_with_codebook,
        "Tokenize n-grams and encode a four-stream frame.");
    module.def(
        "decompress_with_decodebook",
        &decompress_with_decodebook,
        "Decompress a four-stream frame using a Huff0 decodebook.");
    module.def(
        "decompress_ngram_with_decodebook",
        &decompress_ngram_with_decodebook,
        "Decode a four-stream frame and restore packed n-grams.");
    module.def(
        "compress_mixed_streams_many",
        &compress_mixed_streams_many,
        nb::arg("tensor_specs"),
        nb::arg("min_saving_bytes") = 0,
        nb::arg("threads") = 0,
        nb::arg("compression_threshold") = 0.95,
        "Encode many tensors with one mixed-stream Huff0 native job queue.");
    module.def(
        "decompress_mixed_streams_many",
        &decompress_mixed_streams_many,
        nb::arg("tensor_specs"),
        nb::arg("threads") = 0,
        "Decode many mixed-stream tensors with one native job queue.");
}
