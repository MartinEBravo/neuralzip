#define PY_SSIZE_T_CLEAN
#include <Python.h>

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

class OwnedPyRef {
public:
    explicit OwnedPyRef(PyObject* value = nullptr) : value_(value) {}
    ~OwnedPyRef() {
        Py_XDECREF(value_);
    }

    OwnedPyRef(const OwnedPyRef&) = delete;
    OwnedPyRef& operator=(const OwnedPyRef&) = delete;

    PyObject* get() const {
        return value_;
    }

private:
    PyObject* value_;
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
    Py_ssize_t hits = 0;
    Py_ssize_t misses = 0;
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
    std::vector<std::uint64_t> vocab;
    NGramVocabLookup vocab_lookup;
    int ngram_n = 1;
    bool frame_4x = false;
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
    Py_ssize_t hits = 0;
    Py_ssize_t misses = 0;
    int ngram_n = 1;
    std::string error;
};

enum class MixedStreamDecodeMode {
    Raw,
    Zero,
    LocalHuff0,
    SharedHuff0,
};

struct MixedStreamDecodeChunk {
    MixedStreamDecodeMode mode = MixedStreamDecodeMode::Raw;
    Py_buffer payload_view{};
    PyObject* decodebook_capsule = nullptr;
    Huff0Decodebook* decodebook = nullptr;
    std::size_t tensor_index = 0;
    int stream = 0;
    std::size_t output_offset = 0;
    std::size_t symbol_count = 0;
    bool frame_4x = false;
    std::string error;
};

struct MixedStreamDecodeTensor {
    int element_size = 0;
    bool bit_reorder = false;
    std::size_t numel = 0;
    std::vector<std::vector<std::uint8_t>> lanes;
    PyObject* output = nullptr;
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

PyObject* bytes_from_vector(const std::vector<std::uint8_t>& data) {
    const char* pointer = data.empty() ? "" : reinterpret_cast<const char*>(data.data());
    return PyBytes_FromStringAndSize(pointer, static_cast<Py_ssize_t>(data.size()));
}

PyObject* ngram_result_to_tuple(const NGramBlockResult& result) {
    PyObject* payload = bytes_from_vector(result.payload);
    PyObject* symbol_count = PyLong_FromSsize_t(result.symbol_count);
    PyObject* escapes = bytes_from_vector(result.escapes);
    PyObject* tail = bytes_from_vector(result.tail);
    PyObject* hits = PyLong_FromSsize_t(result.hits);
    PyObject* misses = PyLong_FromSsize_t(result.misses);
    if (!payload || !symbol_count || !escapes || !tail || !hits || !misses) {
        Py_XDECREF(payload);
        Py_XDECREF(symbol_count);
        Py_XDECREF(escapes);
        Py_XDECREF(tail);
        Py_XDECREF(hits);
        Py_XDECREF(misses);
        return nullptr;
    }
    PyObject* output = PyTuple_Pack(
        6, payload, symbol_count, escapes, tail, hits, misses);
    Py_DECREF(payload);
    Py_DECREF(symbol_count);
    Py_DECREF(escapes);
    Py_DECREF(tail);
    Py_DECREF(hits);
    Py_DECREF(misses);
    return output;
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

void restore_mixed_stream_tensor_range(
    const MixedStreamDecodeTensor& request,
    std::size_t start,
    std::size_t length) {
    const int element_size = request.element_size;
    const std::size_t end = start + length;
    std::uint8_t* output = request.output_data;
    if (element_size == 2) {
        const auto& first = request.lanes[0];
        const auto& second = request.lanes[1];
        if (!request.bit_reorder) {
            for (std::size_t index = start; index < end; ++index) {
                std::size_t target = index * 2U;
                output[target] = first[index];
                output[target + 1U] = second[index];
            }
            return;
        }
        for (std::size_t index = start; index < end; ++index) {
            const std::uint8_t penultimate = first[index];
            const std::uint8_t last = second[index];
            std::size_t target = index * 2U;
            output[target] = static_cast<std::uint8_t>(
                (penultimate & 0x7FU) | ((last & 0x01U) << 7U));
            output[target + 1U] = static_cast<std::uint8_t>(
                (last >> 1U) | (penultimate & 0x80U));
        }
        return;
    }

    const auto& first = request.lanes[0];
    const auto& second = request.lanes[1];
    const auto& penultimate_lane = request.lanes[2];
    const auto& last_lane = request.lanes[3];
    if (!request.bit_reorder) {
        for (std::size_t index = start; index < end; ++index) {
            std::size_t target = index * 4U;
            output[target] = first[index];
            output[target + 1U] = second[index];
            output[target + 2U] = penultimate_lane[index];
            output[target + 3U] = last_lane[index];
        }
        return;
    }
    for (std::size_t index = start; index < end; ++index) {
        const std::uint8_t penultimate = penultimate_lane[index];
        const std::uint8_t last = last_lane[index];
        std::size_t target = index * 4U;
        output[target] = first[index];
        output[target + 1U] = second[index];
        output[target + 2U] = static_cast<std::uint8_t>(
            (penultimate & 0x7FU) | ((last & 0x01U) << 7U));
        output[target + 3U] = static_cast<std::uint8_t>(
            (last >> 1U) | (penultimate & 0x80U));
    }
}

inline std::uint8_t mixed_stream_decoded_lane_byte(
    const std::uint8_t* lane,
    std::size_t index) {
    return lane == nullptr ? 0U : lane[index];
}

void restore_mixed_stream_aligned_chunk(
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
                output[target] = mixed_stream_decoded_lane_byte(first, index);
                output[target + 1U] = mixed_stream_decoded_lane_byte(second, index);
            }
            return;
        }
        for (std::size_t index = 0; index < length; ++index) {
            const std::uint8_t penultimate = mixed_stream_decoded_lane_byte(
                first, index);
            const std::uint8_t last = mixed_stream_decoded_lane_byte(second, index);
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
            output[target] = mixed_stream_decoded_lane_byte(first, index);
            output[target + 1U] = mixed_stream_decoded_lane_byte(second, index);
            output[target + 2U] = mixed_stream_decoded_lane_byte(
                penultimate_lane, index);
            output[target + 3U] = mixed_stream_decoded_lane_byte(last_lane, index);
        }
        return;
    }
    for (std::size_t index = 0; index < length; ++index) {
        const std::uint8_t penultimate = mixed_stream_decoded_lane_byte(
            penultimate_lane, index);
        const std::uint8_t last = mixed_stream_decoded_lane_byte(last_lane, index);
        const std::size_t target = index * 4U;
        output[target] = mixed_stream_decoded_lane_byte(first, index);
        output[target + 1U] = mixed_stream_decoded_lane_byte(second, index);
        output[target + 2U] = static_cast<std::uint8_t>(
            (penultimate & 0x7FU) | ((last & 0x01U) << 7U));
        output[target + 3U] = static_cast<std::uint8_t>(
            (last >> 1U) | (penultimate & 0x80U));
    }
}

void release_vector(std::vector<std::uint8_t>& data) {
    std::vector<std::uint8_t>().swap(data);
}

std::vector<std::uint8_t> decompress_local_block(
    const std::uint8_t* payload,
    Py_ssize_t payload_len,
    Py_ssize_t symbol_count) {
    if (symbol_count < 0) {
        throw std::runtime_error("symbol_count must be non-negative.");
    }
    std::vector<std::uint8_t> output(static_cast<std::size_t>(symbol_count));
    if (symbol_count == 0) {
        return output;
    }
    std::vector<HUF_DTable> dtable(HUF_DTABLE_SIZE(HUF_TABLELOG_MAX));
    dtable[0] = static_cast<HUF_DTable>(HUF_TABLELOG_MAX * 0x01000001U);
    std::vector<unsigned> workspace(HUF_DECOMPRESS_WORKSPACE_SIZE_U32);
    size_t decoded = HUF_decompress4X_hufOnly_wksp(
        dtable.data(),
        output.data(),
        output.size(),
        payload,
        static_cast<std::size_t>(payload_len),
        workspace.data(),
        workspace.size() * sizeof(unsigned),
        kFlags);
    check_huf(decoded);
    if (decoded != static_cast<std::size_t>(symbol_count)) {
        throw std::runtime_error("Huff0 decoded size mismatch.");
    }
    return output;
}

void decompress_local_block_into(
    const std::uint8_t* payload,
    Py_ssize_t payload_len,
    std::uint8_t* output,
    std::size_t symbol_count) {
    if (symbol_count == 0) {
        return;
    }
    std::vector<HUF_DTable> dtable(HUF_DTABLE_SIZE(HUF_TABLELOG_MAX));
    dtable[0] = static_cast<HUF_DTable>(HUF_TABLELOG_MAX * 0x01000001U);
    std::vector<unsigned> workspace(HUF_DECOMPRESS_WORKSPACE_SIZE_U32);
    size_t decoded = HUF_decompress4X_hufOnly_wksp(
        dtable.data(),
        output,
        symbol_count,
        payload,
        static_cast<std::size_t>(payload_len),
        workspace.data(),
        workspace.size() * sizeof(unsigned),
        kFlags);
    check_huf(decoded);
    if (decoded != symbol_count) {
        throw std::runtime_error("Huff0 decoded size mismatch.");
    }
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
    if (header_len <= 0) {
        throw std::runtime_error("Huff0 codebook header is empty.");
    }
    if (max_bits == 0 || max_bits > HUF_TABLELOG_ABSOLUTEMAX) {
        throw std::runtime_error(
            "max_bits must be in [1, HUF_TABLELOG_ABSOLUTEMAX].");
    }

    std::vector<HUF_CElt> ctable(HUF_CTABLE_SIZE_ST(kMaxSymbolValue));
    unsigned max_symbol_value = kMaxSymbolValue;
    unsigned has_zero_weights = 0;
    const size_t read_size = HUF_readCTable(
        ctable.data(),
        &max_symbol_value,
        header,
        static_cast<std::size_t>(header_len),
        &has_zero_weights);
    check_huf(read_size);
    if (read_size != static_cast<std::size_t>(header_len)) {
        throw std::runtime_error(
            "Huff0 did not consume the full codebook header.");
    }

    const HUF_CTableHeader table_header = HUF_readCTableHeader(ctable.data());
    if (table_header.tableLog > max_bits) {
        throw std::runtime_error(
            "Huff0 codebook exceeds the requested max_bits limit.");
    }

    constexpr unsigned ctable_word_bits = sizeof(HUF_CElt) * 8U;
    ExportedCodeTable output;
    for (unsigned symbol = 0; symbol <= kMaxSymbolValue; ++symbol) {
        const unsigned length = HUF_getNbBitsFromCTable(ctable.data(), symbol);
        if (length > max_bits || length > HUF_TABLELOG_ABSOLUTEMAX) {
            throw std::runtime_error(
                "Huff0 symbol code exceeds the supported bit limit.");
        }
        output.code_lengths[symbol] = static_cast<std::uint8_t>(length);
        output.max_bits = std::max(output.max_bits, length);
        if (length == 0) {
            continue;
        }
        const HUF_CElt entry = ctable[static_cast<std::size_t>(symbol) + 1U];
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
    std::size_t max_output_bytes,
    bool four_streams = false) {
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
    const size_t size = four_streams
        ? HUF_compress4X_usingCTable(
              output.data(), output.size(), data,
              static_cast<std::size_t>(data_len), codebook.ctable.data(), kFlags)
        : HUF_compress1X_usingCTable(
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

std::vector<std::uint8_t> decode_huff0_block(
    const std::uint8_t* payload,
    Py_ssize_t payload_len,
    Py_ssize_t symbol_count,
    const Huff0Decodebook& decodebook,
    bool four_streams = false) {
    if (payload_len < 0) {
        throw std::runtime_error("payload_len must be non-negative.");
    }
    if (symbol_count < 0) {
        throw std::runtime_error("symbol_count must be non-negative.");
    }
    std::vector<std::uint8_t> output(static_cast<std::size_t>(symbol_count));
    if (symbol_count == 0) {
        return output;
    }
    const HUF_DTable* dtable = select_decode_dtable(
        decodebook,
        static_cast<std::size_t>(symbol_count),
        static_cast<std::size_t>(payload_len));
    const size_t decoded = four_streams
        ? HUF_decompress4X_usingDTable(
              output.data(), output.size(), payload,
              static_cast<std::size_t>(payload_len), dtable, kFlags)
        : HUF_decompress1X_usingDTable(
              output.data(), output.size(), payload,
              static_cast<std::size_t>(payload_len), dtable, kFlags);
    check_huf(decoded);
    if (decoded != static_cast<std::size_t>(symbol_count)) {
        throw std::runtime_error("Huff0 decoded size mismatch.");
    }
    return output;
}

void decode_huff0_block_into(
    const std::uint8_t* payload,
    Py_ssize_t payload_len,
    std::uint8_t* output,
    std::size_t symbol_count,
    const Huff0Decodebook& decodebook,
    bool four_streams = false) {
    if (payload_len < 0) {
        throw std::runtime_error("payload_len must be non-negative.");
    }
    if (symbol_count == 0) {
        return;
    }
    const HUF_DTable* dtable = select_decode_dtable(
        decodebook, symbol_count, static_cast<std::size_t>(payload_len));
    const size_t decoded = four_streams
        ? HUF_decompress4X_usingDTable(
              output, symbol_count, payload,
              static_cast<std::size_t>(payload_len), dtable, kFlags)
        : HUF_decompress1X_usingDTable(
              output, symbol_count, payload,
              static_cast<std::size_t>(payload_len), dtable, kFlags);
    check_huf(decoded);
    if (decoded != symbol_count) {
        throw std::runtime_error("Huff0 decoded size mismatch.");
    }
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

std::vector<std::uint64_t> vocab_from_view(const Py_buffer& vocab_view, int n) {
    if (n < 1 || n > 8) {
        throw std::runtime_error("ngram vocab requires n in [1, 8].");
    }
    const Py_ssize_t symbol_bytes = n <= 4 ? 4 : 8;
    if (vocab_view.len % symbol_bytes != 0) {
        throw std::runtime_error(
            "ngram vocab must contain fixed-width little-endian integer values.");
    }
    const auto* data = static_cast<const std::uint8_t*>(vocab_view.buf);
    std::vector<std::uint64_t> vocab(
        static_cast<std::size_t>(vocab_view.len / symbol_bytes));
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
    Py_ssize_t hits = 0;
    Py_ssize_t misses = 0;
};

NGramTokenization tokenize_ngrams(
    const std::uint8_t* data,
    Py_ssize_t data_len,
    int n,
    const std::vector<std::uint64_t>& vocab,
    const NGramVocabLookup* lookup = nullptr) {
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
        std::uint8_t token = 255;
        if (lookup != nullptr) {
            token = find_ngram_token(*lookup, symbol);
        } else {
            auto iterator = std::lower_bound(vocab.begin(), vocab.end(), symbol);
            if (iterator != vocab.end() && *iterator == symbol) {
                std::size_t position = static_cast<std::size_t>(
                    iterator - vocab.begin());
                if (position >= 255) {
                    throw std::runtime_error("ngram vocab id overflow.");
                }
                token = static_cast<std::uint8_t>(position);
            }
        }
        if (token != 255) {
            result.tokens.push_back(token);
            result.hits += 1;
            continue;
        }

        result.tokens.push_back(255);
        result.escapes.insert(result.escapes.end(), pointer, pointer + n);
        result.misses += 1;
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
            result.hits += 1;
            continue;
        }

        result.tokens.push_back(255);
        result.escapes.insert(
            result.escapes.end(), values.begin(), values.begin() + n);
        result.misses += 1;
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
    std::size_t max_output_bytes,
    bool four_streams = false) {
    NGramBlockResult result;
    if (tokens.tokens.empty()) {
        return result;
    }

    result.payload = encode_huff0_block(
        tokens.tokens.data(),
        static_cast<Py_ssize_t>(tokens.tokens.size()),
        codebook,
        max_output_bytes,
        four_streams);
    if (result.payload.empty()
        || result.payload.size() + static_cast<std::size_t>(min_saving_bytes)
            >= tokens.tokens.size()) {
        result.payload.clear();
        return result;
    }

    result.symbol_count = static_cast<Py_ssize_t>(tokens.tokens.size());
    result.escapes = std::move(tokens.escapes);
    result.tail = std::move(tokens.tail);
    result.hits = tokens.hits;
    result.misses = tokens.misses;
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
    bool four_streams = false,
    const NGramVocabLookup* lookup = nullptr) {
    return encode_tokenized_ngram_block(
        tokenize_ngrams(data, data_len, n, vocab, lookup),
        codebook,
        min_saving_bytes,
        max_output_bytes,
        four_streams);
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
                static_cast<std::size_t>(budget),
                spec.frame_4x);
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

    constexpr Py_ssize_t kPackedMetadataBytes = 1 + 8 + 8 + 8 + 8 + 1 + 8 + 8;
    Py_ssize_t budget = std::max<Py_ssize_t>(
        0,
        static_cast<Py_ssize_t>(
            static_cast<double>(result.raw.size()) * compression_threshold)
            - kPackedMetadataBytes);
    NGramBlockResult packed = encode_ngram_block(
        result.raw.data(),
        static_cast<Py_ssize_t>(result.raw.size()),
        spec.ngram_n,
        spec.vocab,
        *spec.codebook,
        min_saving_bytes,
        static_cast<std::size_t>(budget),
        spec.frame_4x,
        &spec.vocab_lookup);
    double compressed_bytes = static_cast<double>(kPackedMetadataBytes)
        + static_cast<double>(packed.payload.size())
        + static_cast<double>(packed.escapes.size())
        + static_cast<double>(packed.tail.size());
    if (
        packed.encoded
        && compressed_bytes
            < static_cast<double>(result.raw.size()) * compression_threshold) {
        result.mode = MixedMode::PackedHuff0;
        result.payload = std::move(packed.payload);
        result.escapes = std::move(packed.escapes);
        result.tail = std::move(packed.tail);
        result.token_count = packed.symbol_count;
        result.hits = packed.hits;
        result.misses = packed.misses;
        result.ngram_n = spec.ngram_n;
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

    constexpr Py_ssize_t kPackedMetadataBytes = 1 + 8 + 8 + 8 + 8 + 1 + 8 + 8;
    Py_ssize_t budget = std::max<Py_ssize_t>(
        0,
        static_cast<Py_ssize_t>(
            static_cast<double>(length) * compression_threshold)
            - kPackedMetadataBytes);
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
        static_cast<std::size_t>(budget),
        spec.frame_4x);
    double compressed_bytes = static_cast<double>(kPackedMetadataBytes)
        + static_cast<double>(packed.payload.size())
        + static_cast<double>(packed.escapes.size())
        + static_cast<double>(packed.tail.size());
    if (
        packed.encoded
        && compressed_bytes
            < static_cast<double>(length) * compression_threshold) {
        result.mode = MixedMode::PackedHuff0;
        result.payload = std::move(packed.payload);
        result.escapes = std::move(packed.escapes);
        result.tail = std::move(packed.tail);
        result.token_count = packed.symbol_count;
        result.hits = packed.hits;
        result.misses = packed.misses;
        result.ngram_n = spec.ngram_n;
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

void destroy_codebook_capsule(PyObject* capsule) {
    void* pointer = PyCapsule_GetPointer(capsule, "neuralzip.Huff0Codebook");
    delete static_cast<Huff0Codebook*>(pointer);
}

Huff0Codebook* codebook_from_capsule(PyObject* capsule) {
    void* pointer = PyCapsule_GetPointer(capsule, "neuralzip.Huff0Codebook");
    if (pointer == nullptr) {
        throw std::runtime_error("Invalid Huff0 codebook capsule.");
    }
    return static_cast<Huff0Codebook*>(pointer);
}

bool parse_mixed_stream_specs(
    PyObject* mode_source,
    int element_size,
    std::size_t base_chunk_size,
    std::vector<MixedStreamSpec>& specs,
    PyObject*& mode_owner) {
    // Take an immutable snapshot so every mode tuple and codebook capsule
    // remains alive while worker threads run without the GIL, even if a raw
    // extension caller supplied a mutable outer sequence.
    PyObject* mode_sequence = PySequence_Tuple(mode_source);
    if (!mode_sequence) {
        return false;
    }
    if (PySequence_Fast_GET_SIZE(mode_sequence) != element_size) {
        Py_DECREF(mode_sequence);
        PyErr_SetString(
            PyExc_ValueError,
            "stream_modes must contain exactly one mode per byte stream.");
        return false;
    }

    try {
        specs.clear();
        specs.resize(static_cast<std::size_t>(element_size));
        for (int stream = 0; stream < element_size; ++stream) {
            PyObject* item = PySequence_Fast_GET_ITEM(mode_sequence, stream);
            const char* kind = nullptr;
            PyObject* capsule = nullptr;
            PyObject* vocab_source = nullptr;
            int ngram_n = 1;
            int frame_4x = 0;
            if (!PyArg_ParseTuple(
                    item,
                    "sOOip",
                    &kind,
                    &capsule,
                    &vocab_source,
                    &ngram_n,
                    &frame_4x)) {
                Py_DECREF(mode_sequence);
                return false;
            }

            MixedStreamSpec& spec = specs[static_cast<std::size_t>(stream)];
            spec.ngram_n = ngram_n;
            spec.frame_4x = frame_4x != 0;
            if (std::strcmp(kind, "raw") == 0) {
                spec.mode = MixedMode::Raw;
                spec.ngram_n = 1;
            } else if (std::strcmp(kind, "shared_huff0") == 0) {
                spec.mode = MixedMode::SharedHuff0;
                spec.codebook = codebook_from_capsule(capsule);
                spec.ngram_n = 1;
            } else if (std::strcmp(kind, "packed_huff0") == 0) {
                if (ngram_n < 2 || ngram_n > 8) {
                    throw std::runtime_error(
                        "packed_huff0 requires ngram_n in [2, 8].");
                }
                spec.mode = MixedMode::PackedHuff0;
                spec.codebook = codebook_from_capsule(capsule);
                Py_buffer vocab_view{};
                if (PyObject_GetBuffer(
                        vocab_source, &vocab_view, PyBUF_CONTIG_RO) < 0) {
                    Py_DECREF(mode_sequence);
                    return false;
                }
                try {
                    spec.vocab = vocab_from_view(vocab_view, ngram_n);
                    PyBuffer_Release(&vocab_view);
                } catch (...) {
                    PyBuffer_Release(&vocab_view);
                    throw;
                }
                if (spec.vocab.empty()) {
                    throw std::runtime_error(
                        "packed_huff0 vocabulary is empty.");
                }
                spec.vocab_lookup = build_ngram_vocab_lookup(spec.vocab);
            } else {
                throw std::runtime_error("Unsupported mixed Huff0 stream mode.");
            }

            if (spec.mode == MixedMode::Raw) {
                spec.chunk_size = base_chunk_size;
            } else if (spec.mode == MixedMode::SharedHuff0) {
                spec.chunk_size = std::min(base_chunk_size, kMaxBlockSize);
            } else {
                std::size_t n = static_cast<std::size_t>(spec.ngram_n);
                std::size_t limit = std::min(base_chunk_size, kMaxBlockSize * n);
                spec.chunk_size = std::max(n, (limit / n) * n);
            }
        }
    } catch (const std::bad_alloc&) {
        Py_DECREF(mode_sequence);
        PyErr_NoMemory();
        return false;
    } catch (const std::exception& error) {
        Py_DECREF(mode_sequence);
        PyErr_SetString(PyExc_ValueError, error.what());
        return false;
    }

    mode_owner = mode_sequence;
    return true;
}

PyObject* mixed_groups_to_python(
    int element_size,
    const std::vector<std::size_t>& group_sizes,
    const std::vector<MixedBlockResult>& results,
    std::size_t result_offset) {
    PyObject* groups = PyList_New(element_size);
    if (!groups) {
        return nullptr;
    }
    std::size_t result_index = result_offset;
    for (int stream = 0; stream < element_size; ++stream) {
        Py_ssize_t group_size = static_cast<Py_ssize_t>(
            group_sizes[static_cast<std::size_t>(stream)]);
        PyObject* chunks = PyList_New(group_size);
        if (!chunks) {
            Py_DECREF(groups);
            return nullptr;
        }
        for (Py_ssize_t index = 0; index < group_size; ++index, ++result_index) {
            if (result_index >= results.size()) {
                Py_DECREF(chunks);
                Py_DECREF(groups);
                PyErr_SetString(
                    PyExc_RuntimeError,
                    "Mixed-stream result index exceeds the native result set.");
                return nullptr;
            }
            const MixedBlockResult& result = results[result_index];
            const auto& body = result.mode == MixedMode::Raw
                ? result.raw
                : result.payload;
            PyObject* tuple = PyTuple_New(9);
            PyObject* mode = PyUnicode_FromString(mixed_mode_name(result.mode));
            PyObject* payload = bytes_from_vector(body);
            PyObject* count = PyLong_FromSsize_t(result.symbol_count);
            PyObject* token_count = nullptr;
            if (result.mode == MixedMode::PackedHuff0) {
                token_count = PyLong_FromSsize_t(result.token_count);
            } else {
                token_count = Py_None;
                Py_INCREF(token_count);
            }
            PyObject* escapes = bytes_from_vector(result.escapes);
            PyObject* tail = bytes_from_vector(result.tail);
            PyObject* ngram_n = PyLong_FromLong(result.ngram_n);
            PyObject* hits = PyLong_FromSsize_t(result.hits);
            PyObject* misses = PyLong_FromSsize_t(result.misses);
            if (
                !tuple || !mode || !payload || !count || !token_count
                || !escapes || !tail || !ngram_n || !hits || !misses) {
                Py_XDECREF(tuple);
                Py_XDECREF(mode);
                Py_XDECREF(payload);
                Py_XDECREF(count);
                Py_XDECREF(token_count);
                Py_XDECREF(escapes);
                Py_XDECREF(tail);
                Py_XDECREF(ngram_n);
                Py_XDECREF(hits);
                Py_XDECREF(misses);
                Py_DECREF(chunks);
                Py_DECREF(groups);
                return nullptr;
            }
            PyTuple_SET_ITEM(tuple, 0, mode);
            PyTuple_SET_ITEM(tuple, 1, payload);
            PyTuple_SET_ITEM(tuple, 2, count);
            PyTuple_SET_ITEM(tuple, 3, token_count);
            PyTuple_SET_ITEM(tuple, 4, escapes);
            PyTuple_SET_ITEM(tuple, 5, tail);
            PyTuple_SET_ITEM(tuple, 6, ngram_n);
            PyTuple_SET_ITEM(tuple, 7, hits);
            PyTuple_SET_ITEM(tuple, 8, misses);
            PyList_SET_ITEM(chunks, index, tuple);
        }
        PyList_SET_ITEM(groups, stream, chunks);
    }
    return groups;
}

void destroy_decodebook_capsule(PyObject* capsule) {
    void* pointer = PyCapsule_GetPointer(capsule, "neuralzip.Huff0Decodebook");
    delete static_cast<Huff0Decodebook*>(pointer);
}

Huff0Decodebook* decodebook_from_capsule(PyObject* capsule) {
    void* pointer = PyCapsule_GetPointer(capsule, "neuralzip.Huff0Decodebook");
    if (pointer == nullptr) {
        throw std::runtime_error("Invalid Huff0 decodebook capsule.");
    }
    return static_cast<Huff0Decodebook*>(pointer);
}

PyObject* codebook_to_python(std::unique_ptr<Huff0Codebook> codebook) {
    Huff0Codebook* native = codebook.get();
    PyObject* capsule = PyCapsule_New(
        native,
        "neuralzip.Huff0Codebook",
        destroy_codebook_capsule);
    if (!capsule) {
        return nullptr;
    }
    codebook.release();
    PyObject* header = bytes_from_vector(native->serialized);
    if (!header) {
        Py_DECREF(capsule);
        return nullptr;
    }
    PyObject* result = PyTuple_Pack(2, capsule, header);
    Py_DECREF(capsule);
    Py_DECREF(header);
    return result;
}

bool validate_codebook_options(Py_ssize_t prior_count, Py_ssize_t max_bits) {
    if (prior_count < 0 || prior_count > static_cast<Py_ssize_t>(
            std::numeric_limits<unsigned>::max())) {
        PyErr_SetString(PyExc_ValueError, "prior_count must fit in unsigned.");
        return false;
    }
    if (max_bits < 1 || max_bits > static_cast<Py_ssize_t>(kMaxTableLog)) {
        PyErr_SetString(
            PyExc_ValueError,
            "max_bits must be in [1, 12].");
        return false;
    }
    return true;
}

bool parse_symbol_counts(
    PyObject* source,
    unsigned prior_count,
    std::array<unsigned, 256>& output) {
    OwnedPyRef sequence(PySequence_Fast(
        source, "symbol_counts must be a sequence of 256 integers."));
    if (!sequence.get()) {
        return false;
    }
    if (PySequence_Fast_GET_SIZE(sequence.get()) != 256) {
        PyErr_SetString(
            PyExc_ValueError,
            "symbol_counts must contain exactly 256 entries.");
        return false;
    }

    std::array<std::uint64_t, 256> counts{};
    std::uint64_t total = 0;
    for (Py_ssize_t index = 0; index < 256; ++index) {
        PyObject* item = PySequence_Fast_GET_ITEM(sequence.get(), index);
        const unsigned long long count = PyLong_AsUnsignedLongLong(item);
        if (PyErr_Occurred()) {
            return false;
        }
        if (total > std::numeric_limits<std::uint64_t>::max() - count) {
            PyErr_SetString(PyExc_OverflowError, "Huff0 symbol count overflow.");
            return false;
        }
        counts[static_cast<std::size_t>(index)] = count;
        total += count;
    }

    const std::uint64_t prior_total =
        static_cast<std::uint64_t>(prior_count) * counts.size();
    if (prior_total >= kMaxHistogramTotal) {
        PyErr_SetString(
            PyExc_ValueError,
            "prior_count is too large for a 256-symbol Huff0 table.");
        return false;
    }
    const std::uint64_t available =
        kMaxHistogramTotal - prior_total;

    unsigned shift = 0;
    if (total > available) {
        for (shift = 1; shift < 64; ++shift) {
            std::uint64_t scaled_total = 0;
            for (const std::uint64_t count : counts) {
                if (count == 0) {
                    continue;
                }
                scaled_total += std::max<std::uint64_t>(1, count >> shift);
            }
            if (scaled_total <= available) {
                break;
            }
        }
        if (shift == 64) {
            PyErr_SetString(PyExc_OverflowError, "Huff0 symbol count overflow.");
            return false;
        }
    }

    for (std::size_t index = 0; index < counts.size(); ++index) {
        const std::uint64_t count = counts[index];
        const std::uint64_t scaled =
            count == 0 ? 0 : std::max<std::uint64_t>(1, count >> shift);
        output[index] = static_cast<unsigned>(scaled + prior_count);
    }
    return true;
}

PyObject* py_max_block_size(PyObject*, PyObject*) {
    return PyLong_FromSize_t(kMaxBlockSize);
}

PyObject* py_build_codebook(PyObject*, PyObject* args, PyObject* kwargs) {
    Py_buffer data_view;
    Py_ssize_t prior_count = 0;
    Py_ssize_t max_bits = kTableLog;
    static const char* keyword_names[] = {
        "data", "prior_count", "max_bits", nullptr};

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "y*|nn",
            const_cast<char**>(keyword_names),
            &data_view,
            &prior_count,
            &max_bits)) {
        return nullptr;
    }
    if (!validate_codebook_options(prior_count, max_bits)) {
        PyBuffer_Release(&data_view);
        return nullptr;
    }

    try {
        const auto* data = static_cast<const std::uint8_t*>(data_view.buf);
        auto codebook = make_codebook(
            data,
            data_view.len,
            static_cast<unsigned>(prior_count),
            static_cast<unsigned>(max_bits));
        PyBuffer_Release(&data_view);
        return codebook_to_python(std::move(codebook));
    } catch (const std::bad_alloc&) {
        PyBuffer_Release(&data_view);
        PyErr_NoMemory();
        return nullptr;
    } catch (const std::exception& error) {
        PyBuffer_Release(&data_view);
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

PyObject* py_build_codebook_from_counts(
    PyObject*, PyObject* args, PyObject* kwargs) {
    PyObject* count_source = nullptr;
    Py_ssize_t prior_count = 0;
    Py_ssize_t max_bits = kTableLog;
    static const char* keyword_names[] = {
        "symbol_counts", "prior_count", "max_bits", nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "O|nn",
            const_cast<char**>(keyword_names),
            &count_source,
            &prior_count,
            &max_bits)) {
        return nullptr;
    }
    if (!validate_codebook_options(prior_count, max_bits)) {
        return nullptr;
    }

    std::array<unsigned, 256> counts{};
    if (!parse_symbol_counts(
            count_source, static_cast<unsigned>(prior_count), counts)) {
        return nullptr;
    }
    try {
        return codebook_to_python(make_codebook_from_counts(
            counts, static_cast<unsigned>(max_bits)));
    } catch (const std::bad_alloc&) {
        PyErr_NoMemory();
        return nullptr;
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

PyObject* py_build_codebook_from_header(PyObject*, PyObject* args) {
    Py_buffer header_view;
    if (!PyArg_ParseTuple(args, "y*", &header_view)) {
        return nullptr;
    }
    try {
        const auto* header = static_cast<const std::uint8_t*>(header_view.buf);
        auto codebook = make_codebook_from_header(header, header_view.len);
        PyBuffer_Release(&header_view);
        return codebook_to_python(std::move(codebook));
    } catch (const std::bad_alloc&) {
        PyBuffer_Release(&header_view);
        PyErr_NoMemory();
        return nullptr;
    } catch (const std::exception& error) {
        PyBuffer_Release(&header_view);
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

PyObject* py_build_decodebook(PyObject*, PyObject* args) {
    Py_buffer header_view;
    if (!PyArg_ParseTuple(args, "y*", &header_view)) {
        return nullptr;
    }
    try {
        const auto* header = static_cast<const std::uint8_t*>(header_view.buf);
        auto decodebook = make_decodebook_from_header(header, header_view.len);
        PyObject* capsule = PyCapsule_New(
            decodebook.get(),
            "neuralzip.Huff0Decodebook",
            destroy_decodebook_capsule);
        PyBuffer_Release(&header_view);
        if (!capsule) {
            return nullptr;
        }
        decodebook.release();
        return capsule;
    } catch (const std::bad_alloc&) {
        PyBuffer_Release(&header_view);
        PyErr_NoMemory();
        return nullptr;
    } catch (const std::exception& error) {
        PyBuffer_Release(&header_view);
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

PyObject* py_export_code_table(
    PyObject*, PyObject* args, PyObject* kwargs) {
    Py_buffer header_view;
    Py_ssize_t max_bits = 12;
    static const char* keyword_names[] = {
        "codebook", "max_bits", nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "y*|n",
            const_cast<char**>(keyword_names),
            &header_view,
            &max_bits)) {
        return nullptr;
    }
    if (
        max_bits <= 0
        || max_bits > 12) {
        PyBuffer_Release(&header_view);
        PyErr_Format(
            PyExc_ValueError,
            "max_bits must be in [1, 12].");
        return nullptr;
    }

    try {
        const auto* header = static_cast<const std::uint8_t*>(header_view.buf);
        const ExportedCodeTable table = export_code_table_from_header(
            header, header_view.len, static_cast<unsigned>(max_bits));
        PyBuffer_Release(&header_view);

        PyObject* lengths = PyBytes_FromStringAndSize(
            reinterpret_cast<const char*>(table.code_lengths.data()),
            static_cast<Py_ssize_t>(table.code_lengths.size()));
        PyObject* values = PyTuple_New(
            static_cast<Py_ssize_t>(table.code_values.size()));
        PyObject* actual_max_bits = PyLong_FromUnsignedLong(table.max_bits);
        if (!lengths || !values || !actual_max_bits) {
            Py_XDECREF(lengths);
            Py_XDECREF(values);
            Py_XDECREF(actual_max_bits);
            return nullptr;
        }
        for (std::size_t index = 0; index < table.code_values.size(); ++index) {
            PyObject* value = PyLong_FromUnsignedLong(table.code_values[index]);
            if (!value) {
                Py_DECREF(lengths);
                Py_DECREF(values);
                Py_DECREF(actual_max_bits);
                return nullptr;
            }
            PyTuple_SET_ITEM(values, static_cast<Py_ssize_t>(index), value);
        }

        PyObject* result = PyTuple_Pack(
            3, lengths, values, actual_max_bits);
        Py_DECREF(lengths);
        Py_DECREF(values);
        Py_DECREF(actual_max_bits);
        return result;
    } catch (const std::bad_alloc&) {
        PyBuffer_Release(&header_view);
        PyErr_NoMemory();
        return nullptr;
    } catch (const std::exception& error) {
        PyBuffer_Release(&header_view);
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

PyObject* py_encode_gpu_bfloat16(
    PyObject*, PyObject* args, PyObject* kwargs) {
    Py_buffer raw_view{};
    Py_buffer header_view{};
    Py_ssize_t bytes_per_thread = 8;
    Py_ssize_t threads_per_block = 512;
    static const char* keyword_names[] = {
        "raw_bf16",
        "codebook",
        "bytes_per_thread",
        "threads_per_block",
        nullptr,
    };
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "y*y*|nn",
            const_cast<char**>(keyword_names),
            &raw_view,
            &header_view,
            &bytes_per_thread,
            &threads_per_block)) {
        return nullptr;
    }

    auto release_buffers = [&]() {
        PyBuffer_Release(&raw_view);
        PyBuffer_Release(&header_view);
    };
    if (bytes_per_thread <= 0 || threads_per_block <= 0) {
        release_buffers();
        PyErr_SetString(
            PyExc_ValueError,
            "bytes_per_thread and threads_per_block must be positive.");
        return nullptr;
    }
    if (bytes_per_thread != 8) {
        release_buffers();
        PyErr_SetString(
            PyExc_ValueError,
            "bytes_per_thread must be 8 for this ABI.");
        return nullptr;
    }
    if (threads_per_block != 512) {
        release_buffers();
        PyErr_SetString(
            PyExc_ValueError,
            "threads_per_block must be 512 for this ABI.");
        return nullptr;
    }
    if (
        static_cast<std::uint64_t>(bytes_per_thread)
            > std::numeric_limits<std::uint32_t>::max()
        || static_cast<std::uint64_t>(threads_per_block)
            > std::numeric_limits<std::uint32_t>::max()) {
        release_buffers();
        PyErr_SetString(
            PyExc_OverflowError,
            "GPU layout dimensions must fit in uint32.");
        return nullptr;
    }
    if ((raw_view.len & 1) != 0) {
        release_buffers();
        PyErr_SetString(
            PyExc_ValueError,
            "A BF16 byte sequence must have an even length.");
        return nullptr;
    }

    const std::uint8_t* raw = static_cast<const std::uint8_t*>(raw_view.buf);
    const std::uint8_t* header =
        static_cast<const std::uint8_t*>(header_view.buf);
    std::vector<std::uint8_t> raw_snapshot;
    std::vector<std::uint8_t> header_snapshot;
    try {
        // PyBUF_CONTIG_RO can still expose mutable bytearray/memoryview storage.
        // Snapshot such inputs before releasing the GIL so another Python thread
        // cannot mutate bytes while the native encoder is reading them.
        if (!raw_view.readonly && raw_view.len > 0) {
            raw_snapshot.assign(raw, raw + raw_view.len);
            raw = raw_snapshot.data();
        }
        if (!header_view.readonly && header_view.len > 0) {
            header_snapshot.assign(header, header + header_view.len);
            header = header_snapshot.data();
        }
    } catch (const std::bad_alloc&) {
        release_buffers();
        PyErr_NoMemory();
        return nullptr;
    }

    GPUBFloat16Encoding encoded;
    std::exception_ptr failure;
    PyThreadState* saved = PyEval_SaveThread();
    try {
        encoded = encode_gpu_bfloat16_layout(
            raw,
            static_cast<std::size_t>(raw_view.len),
            header,
            header_view.len,
            static_cast<std::uint32_t>(bytes_per_thread),
            static_cast<std::uint32_t>(threads_per_block));
    } catch (...) {
        failure = std::current_exception();
    }
    PyEval_RestoreThread(saved);
    release_buffers();

    if (failure) {
        try {
            std::rethrow_exception(failure);
        } catch (const std::bad_alloc&) {
            PyErr_NoMemory();
        } catch (const std::overflow_error& error) {
            PyErr_SetString(PyExc_OverflowError, error.what());
        } catch (const std::exception& error) {
            PyErr_SetString(PyExc_ValueError, error.what());
        } catch (...) {
            PyErr_SetString(
                PyExc_RuntimeError, "Unknown GPU-layout encoding failure.");
        }
        return nullptr;
    }

    PyObject* exponent_bytes = bytes_from_vector(encoded.encoded_exponents);
    PyObject* sign_mantissa = bytes_from_vector(encoded.sign_mantissa);
    PyObject* packed_gaps = bytes_from_vector(encoded.packed_gaps);
    PyObject* positions = PyTuple_New(
        static_cast<Py_ssize_t>(encoded.output_positions.size()));
    PyObject* bit_length = PyLong_FromUnsignedLongLong(encoded.bit_length);
    PyObject* max_bits = PyLong_FromUnsignedLong(encoded.max_bits);
    if (
        !exponent_bytes || !sign_mantissa || !packed_gaps || !positions
        || !bit_length || !max_bits) {
        Py_XDECREF(exponent_bytes);
        Py_XDECREF(sign_mantissa);
        Py_XDECREF(packed_gaps);
        Py_XDECREF(positions);
        Py_XDECREF(bit_length);
        Py_XDECREF(max_bits);
        return nullptr;
    }
    for (std::size_t index = 0; index < encoded.output_positions.size(); ++index) {
        PyObject* position = PyLong_FromUnsignedLong(
            encoded.output_positions[index]);
        if (!position) {
            Py_DECREF(exponent_bytes);
            Py_DECREF(sign_mantissa);
            Py_DECREF(packed_gaps);
            Py_DECREF(positions);
            Py_DECREF(bit_length);
            Py_DECREF(max_bits);
            return nullptr;
        }
        PyTuple_SET_ITEM(positions, static_cast<Py_ssize_t>(index), position);
    }

    PyObject* result = PyTuple_Pack(
        6,
        exponent_bytes,
        sign_mantissa,
        packed_gaps,
        positions,
        bit_length,
        max_bits);
    Py_DECREF(exponent_bytes);
    Py_DECREF(sign_mantissa);
    Py_DECREF(packed_gaps);
    Py_DECREF(positions);
    Py_DECREF(bit_length);
    Py_DECREF(max_bits);
    return result;
}

PyObject* py_encode_with_codebook_impl(
    PyObject* args, PyObject* kwargs, bool four_streams) {
    Py_buffer data_view;
    PyObject* capsule = nullptr;
    Py_ssize_t min_saving_bytes = 0;
    Py_ssize_t max_output_bytes = -1;
    static const char* keyword_names[] = {
        "data",
        "codebook",
        "min_saving_bytes",
        "max_output_bytes",
        nullptr,
    };

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "y*O|nn",
            const_cast<char**>(keyword_names),
            &data_view,
            &capsule,
            &min_saving_bytes,
            &max_output_bytes)) {
        return nullptr;
    }
    if (min_saving_bytes < 0) {
        PyBuffer_Release(&data_view);
        PyErr_SetString(PyExc_ValueError, "min_saving_bytes must be non-negative.");
        return nullptr;
    }

    try {
        Huff0Codebook* codebook = codebook_from_capsule(capsule);
        const auto* data = static_cast<const std::uint8_t*>(data_view.buf);
        std::size_t max_bytes =
            max_output_bytes < 0
                ? std::numeric_limits<std::size_t>::max()
                : static_cast<std::size_t>(max_output_bytes);
        std::vector<std::uint8_t> encoded =
            encode_huff0_block(
                data,
                data_view.len,
                *codebook,
                max_bytes,
                four_streams);

        if (encoded.empty()
            || encoded.size() + static_cast<std::size_t>(min_saving_bytes)
                >= static_cast<std::size_t>(data_view.len)) {
            PyBuffer_Release(&data_view);
            Py_RETURN_NONE;
        }

        PyObject* payload = bytes_from_vector(encoded);
        PyObject* symbol_count = PyLong_FromSsize_t(data_view.len);
        PyBuffer_Release(&data_view);
        if (!payload || !symbol_count) {
            Py_XDECREF(payload);
            Py_XDECREF(symbol_count);
            return nullptr;
        }
        PyObject* result = PyTuple_Pack(2, payload, symbol_count);
        Py_DECREF(payload);
        Py_DECREF(symbol_count);
        return result;
    } catch (const std::bad_alloc&) {
        PyBuffer_Release(&data_view);
        PyErr_NoMemory();
        return nullptr;
    } catch (const std::exception& error) {
        PyBuffer_Release(&data_view);
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

PyObject* py_encode_with_codebook(PyObject*, PyObject* args, PyObject* kwargs) {
    return py_encode_with_codebook_impl(args, kwargs, false);
}

PyObject* py_encode_with_codebook_4x(PyObject*, PyObject* args, PyObject* kwargs) {
    return py_encode_with_codebook_impl(args, kwargs, true);
}

PyObject* py_encode_ngram_with_codebook_impl(
    PyObject* args, PyObject* kwargs, bool four_streams) {
    Py_buffer data_view;
    PyObject* capsule = nullptr;
    Py_buffer vocab_view;
    Py_ssize_t n = 0;
    Py_ssize_t min_saving_bytes = 0;
    Py_ssize_t max_output_bytes = -1;
    static const char* keyword_names[] = {
        "data",
        "codebook",
        "vocab",
        "n",
        "min_saving_bytes",
        "max_output_bytes",
        nullptr,
    };

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "y*Oy*n|nn",
            const_cast<char**>(keyword_names),
            &data_view,
            &capsule,
            &vocab_view,
            &n,
            &min_saving_bytes,
            &max_output_bytes)) {
        return nullptr;
    }
    if (n < 1 || n > 8) {
        PyBuffer_Release(&data_view);
        PyBuffer_Release(&vocab_view);
        PyErr_SetString(PyExc_ValueError, "n must be in [1, 8].");
        return nullptr;
    }
    if (min_saving_bytes < 0) {
        PyBuffer_Release(&data_view);
        PyBuffer_Release(&vocab_view);
        PyErr_SetString(PyExc_ValueError, "min_saving_bytes must be non-negative.");
        return nullptr;
    }

    try {
        Huff0Codebook* codebook = codebook_from_capsule(capsule);
        const auto* data = static_cast<const std::uint8_t*>(data_view.buf);
        std::vector<std::uint64_t> vocab = vocab_from_view(
            vocab_view, static_cast<int>(n));
        NGramVocabLookup vocab_lookup = build_ngram_vocab_lookup(vocab);
        std::size_t max_bytes =
            max_output_bytes < 0
                ? std::numeric_limits<std::size_t>::max()
                : static_cast<std::size_t>(max_output_bytes);
        NGramBlockResult encoded = encode_ngram_block(
            data,
            data_view.len,
            static_cast<int>(n),
            vocab,
            *codebook,
            min_saving_bytes,
            max_bytes,
            four_streams,
            &vocab_lookup);
        if (!encoded.encoded) {
            PyBuffer_Release(&data_view);
            PyBuffer_Release(&vocab_view);
            Py_RETURN_NONE;
        }
        PyBuffer_Release(&data_view);
        PyBuffer_Release(&vocab_view);
        return ngram_result_to_tuple(encoded);
    } catch (const std::bad_alloc&) {
        PyBuffer_Release(&data_view);
        PyBuffer_Release(&vocab_view);
        PyErr_NoMemory();
        return nullptr;
    } catch (const std::exception& error) {
        PyBuffer_Release(&data_view);
        PyBuffer_Release(&vocab_view);
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

PyObject* py_encode_ngram_with_codebook(
    PyObject*, PyObject* args, PyObject* kwargs) {
    return py_encode_ngram_with_codebook_impl(args, kwargs, false);
}

PyObject* py_encode_ngram_with_codebook_4x(
    PyObject*, PyObject* args, PyObject* kwargs) {
    return py_encode_ngram_with_codebook_impl(args, kwargs, true);
}

PyObject* py_decompress_with_decodebook_impl(PyObject* args, bool four_streams) {
    Py_buffer payload_view;
    PyObject* capsule = nullptr;
    Py_ssize_t symbol_count;
    if (!PyArg_ParseTuple(args, "y*On", &payload_view, &capsule, &symbol_count)) {
        return nullptr;
    }

    try {
        Huff0Decodebook* decodebook = decodebook_from_capsule(capsule);
        const auto* payload = static_cast<const std::uint8_t*>(payload_view.buf);
        std::vector<std::uint8_t> decoded =
            decode_huff0_block(
                payload,
                payload_view.len,
                symbol_count,
                *decodebook,
                four_streams);
        PyObject* result = bytes_from_vector(decoded);
        PyBuffer_Release(&payload_view);
        return result;
    } catch (const std::bad_alloc&) {
        PyBuffer_Release(&payload_view);
        PyErr_NoMemory();
        return nullptr;
    } catch (const std::exception& error) {
        PyBuffer_Release(&payload_view);
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

PyObject* py_decompress_with_decodebook(PyObject*, PyObject* args) {
    return py_decompress_with_decodebook_impl(args, false);
}

PyObject* py_decompress_with_decodebook_4x(PyObject*, PyObject* args) {
    return py_decompress_with_decodebook_impl(args, true);
}

PyObject* py_decompress_ngram_with_decodebook_impl(
    PyObject* args, bool four_streams) {
    Py_buffer payload_view{};
    PyObject* capsule = nullptr;
    Py_ssize_t symbol_count = 0;
    Py_buffer vocab_view{};
    Py_ssize_t n = 0;
    Py_buffer escape_view{};
    Py_buffer tail_view{};
    if (!PyArg_ParseTuple(
            args,
            "y*Ony*ny*y*",
            &payload_view,
            &capsule,
            &symbol_count,
            &vocab_view,
            &n,
            &escape_view,
            &tail_view)) {
        return nullptr;
    }

    auto release_buffers = [&]() {
        PyBuffer_Release(&payload_view);
        PyBuffer_Release(&vocab_view);
        PyBuffer_Release(&escape_view);
        PyBuffer_Release(&tail_view);
    };

    try {
        Huff0Decodebook* decodebook = decodebook_from_capsule(capsule);
        const auto* payload = static_cast<const std::uint8_t*>(payload_view.buf);
        std::vector<std::uint8_t> tokens = decode_huff0_block(
            payload,
            payload_view.len,
            symbol_count,
            *decodebook,
            four_streams);
        std::vector<std::uint64_t> vocab = vocab_from_view(
            vocab_view, static_cast<int>(n));
        std::vector<std::uint8_t> decoded = restore_ngram_tokens(
            tokens,
            static_cast<int>(n),
            vocab,
            static_cast<const std::uint8_t*>(escape_view.buf),
            escape_view.len,
            static_cast<const std::uint8_t*>(tail_view.buf),
            tail_view.len);
        PyObject* result = bytes_from_vector(decoded);
        release_buffers();
        return result;
    } catch (const std::bad_alloc&) {
        release_buffers();
        PyErr_NoMemory();
        return nullptr;
    } catch (const std::exception& error) {
        release_buffers();
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

PyObject* py_decompress_ngram_with_decodebook(PyObject*, PyObject* args) {
    return py_decompress_ngram_with_decodebook_impl(args, false);
}

PyObject* py_decompress_ngram_with_decodebook_4x(PyObject*, PyObject* args) {
    return py_decompress_ngram_with_decodebook_impl(args, true);
}

PyObject* py_compress_mixed_streams_many(
    PyObject*, PyObject* args, PyObject* kwargs) {
    PyObject* tensor_source = nullptr;
    Py_ssize_t min_saving_bytes = 0;
    int requested_threads = 0;
    double compression_threshold = 0.95;
    static const char* keyword_names[] = {
        "tensor_specs",
        "min_saving_bytes",
        "threads",
        "compression_threshold",
        nullptr,
    };
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "O|nid",
            const_cast<char**>(keyword_names),
            &tensor_source,
            &min_saving_bytes,
            &requested_threads,
            &compression_threshold)) {
        return nullptr;
    }
    if (min_saving_bytes < 0 || requested_threads < 0) {
        PyErr_SetString(
            PyExc_ValueError,
            "min_saving_bytes and threads must be non-negative.");
        return nullptr;
    }
    if (!(compression_threshold > 0.0 && compression_threshold <= 1.0)) {
        PyErr_SetString(
            PyExc_ValueError,
            "compression_threshold must be in (0, 1].");
        return nullptr;
    }

    PyObject* tensor_sequence = PySequence_Fast(
        tensor_source, "tensor_specs must be a sequence");
    if (!tensor_sequence) {
        return nullptr;
    }

    struct MixedTensorRequest {
        Py_buffer raw_view{};
        int element_size = 0;
        bool bit_reorder = false;
        PyObject* mode_owner = nullptr;
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

    Py_ssize_t tensor_count = PySequence_Fast_GET_SIZE(tensor_sequence);
    std::vector<MixedTensorRequest> requests;
    std::vector<MixedModelJob> jobs;
    try {
        requests.resize(static_cast<std::size_t>(tensor_count));
    } catch (const std::bad_alloc&) {
        Py_DECREF(tensor_sequence);
        PyErr_NoMemory();
        return nullptr;
    }
    auto release_request_buffers = [&]() {
        for (auto& request : requests) {
            if (request.raw_view.obj != nullptr) {
                PyBuffer_Release(&request.raw_view);
            }
            Py_XDECREF(request.mode_owner);
            request.mode_owner = nullptr;
        }
    };
    auto cleanup_and_error = [&](const char* message) -> PyObject* {
        release_request_buffers();
        Py_DECREF(tensor_sequence);
        if (message != nullptr && !PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError, message);
        }
        return nullptr;
    };

    std::size_t total_input_bytes = 0;
    for (Py_ssize_t tensor_index = 0; tensor_index < tensor_count; ++tensor_index) {
        PyObject* item = PySequence_Fast_GET_ITEM(tensor_sequence, tensor_index);
        PyObject* raw_source = nullptr;
        int element_size = 0;
        int bit_reorder = 0;
        Py_ssize_t requested_chunk_size = 0;
        PyObject* mode_source = nullptr;
        if (!PyArg_ParseTuple(
                item,
                "OiinO",
                &raw_source,
                &element_size,
                &bit_reorder,
                &requested_chunk_size,
                &mode_source)) {
            return cleanup_and_error(nullptr);
        }
        if (element_size != 2 && element_size != 4) {
            return cleanup_and_error("element_size must be 2 or 4.");
        }
        if (requested_chunk_size <= 0) {
            return cleanup_and_error("chunk_size must be positive.");
        }

        MixedTensorRequest& request = requests[static_cast<std::size_t>(tensor_index)];
        if (PyObject_GetBuffer(
                raw_source, &request.raw_view, PyBUF_CONTIG_RO) < 0) {
            return cleanup_and_error(nullptr);
        }
        if (request.raw_view.len % element_size != 0) {
            return cleanup_and_error(
                "data length must be divisible by element_size.");
        }
        std::size_t raw_bytes = static_cast<std::size_t>(request.raw_view.len);
        if (raw_bytes > std::numeric_limits<std::size_t>::max() - total_input_bytes) {
            PyErr_SetString(PyExc_OverflowError, "Total mixed input size overflowed.");
            return cleanup_and_error(nullptr);
        }
        total_input_bytes += raw_bytes;
        request.element_size = element_size;
        request.bit_reorder = bit_reorder != 0;
        if (!parse_mixed_stream_specs(
                mode_source,
                element_size,
                static_cast<std::size_t>(requested_chunk_size),
                request.specs,
                request.mode_owner)) {
            return cleanup_and_error(nullptr);
        }

        try {
            request.group_sizes.assign(
                static_cast<std::size_t>(element_size), 0);
            request.result_offset = jobs.size();
            std::size_t symbol_count = raw_bytes
                / static_cast<std::size_t>(element_size);
            for (int stream = 0; stream < element_size; ++stream) {
                std::size_t step = request.specs[
                    static_cast<std::size_t>(stream)].chunk_size;
                for (std::size_t start = 0; start < symbol_count; start += step) {
                    jobs.push_back(
                        {
                            static_cast<std::size_t>(tensor_index),
                            stream,
                            start,
                            std::min(step, symbol_count - start),
                        });
                    request.group_sizes[static_cast<std::size_t>(stream)] += 1;
                }
            }
        } catch (const std::bad_alloc&) {
            PyErr_NoMemory();
            return cleanup_and_error(nullptr);
        }
    }

    std::vector<MixedBlockResult> results;
    try {
        results.resize(jobs.size());
    } catch (const std::bad_alloc&) {
        PyErr_NoMemory();
        return cleanup_and_error(nullptr);
    }

    PyThreadState* saved = PyEval_SaveThread();
    std::string thread_error = run_parallel_jobs(
        jobs.size(),
        total_input_bytes,
        requested_threads,
        [&](std::size_t job_index) {
            const MixedModelJob& job = jobs[job_index];
            const MixedTensorRequest& request = requests[job.tensor_index];
            const MixedStreamSpec& spec = request.specs[
                static_cast<std::size_t>(job.stream)];
            MixedBlockResult& result = results[job_index];
            try {
                process_mixed_stream_block(
                    result,
                    spec,
                    static_cast<const std::uint8_t*>(request.raw_view.buf),
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
                result.error = "Unknown model-wide mixed encoding failure.";
            }
        });
    PyEval_RestoreThread(saved);

    release_request_buffers();
    Py_DECREF(tensor_sequence);
    if (!thread_error.empty()) {
        PyErr_SetString(PyExc_RuntimeError, thread_error.c_str());
        return nullptr;
    }
    for (const auto& result : results) {
        if (!result.error.empty()) {
            PyErr_SetString(PyExc_ValueError, result.error.c_str());
            return nullptr;
        }
    }

    PyObject* output = PyList_New(tensor_count);
    if (!output) {
        return nullptr;
    }
    for (Py_ssize_t tensor_index = 0; tensor_index < tensor_count; ++tensor_index) {
        const MixedTensorRequest& request = requests[
            static_cast<std::size_t>(tensor_index)];
        PyObject* groups = mixed_groups_to_python(
            request.element_size,
            request.group_sizes,
            results,
            request.result_offset);
        if (!groups) {
            Py_DECREF(output);
            return nullptr;
        }
        PyList_SET_ITEM(output, tensor_index, groups);
    }
    return output;
}

PyObject* py_decompress_mixed_streams_many(
    PyObject*, PyObject* args, PyObject* kwargs) {
    PyObject* tensor_source = nullptr;
    int requested_threads = 0;
    static const char* keyword_names[] = {
        "tensor_specs", "threads", nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "O|i",
            const_cast<char**>(keyword_names),
            &tensor_source,
            &requested_threads)) {
        return nullptr;
    }
    if (requested_threads < 0) {
        PyErr_SetString(PyExc_ValueError, "threads must be non-negative.");
        return nullptr;
    }

    OwnedPyRef tensor_sequence(
        PySequence_Fast(tensor_source, "tensor_specs must be a sequence"));
    if (!tensor_sequence.get()) {
        return nullptr;
    }

    const Py_ssize_t tensor_count = PySequence_Fast_GET_SIZE(
        tensor_sequence.get());
    std::vector<MixedStreamDecodeTensor> requests;
    std::vector<MixedStreamDecodeChunk> chunks;
    std::vector<MixedStreamAlignedDecodeJob> aligned_jobs;
    std::vector<std::size_t> fallback_chunk_jobs;
    std::vector<MixedStreamCombineJob> combine_jobs;
    try {
        requests.resize(static_cast<std::size_t>(tensor_count));
    } catch (const std::bad_alloc&) {
        PyErr_NoMemory();
        return nullptr;
    }

    auto release_inputs = [&]() {
        for (auto& chunk : chunks) {
            if (chunk.payload_view.obj != nullptr) {
                PyBuffer_Release(&chunk.payload_view);
                chunk.payload_view.obj = nullptr;
            }
            Py_XDECREF(chunk.decodebook_capsule);
            chunk.decodebook_capsule = nullptr;
            chunk.decodebook = nullptr;
        }
    };
    auto release_outputs = [&]() {
        for (auto& request : requests) {
            Py_XDECREF(request.output);
            request.output = nullptr;
            request.output_data = nullptr;
        }
    };
    auto fail = [&](const char* message) -> PyObject* {
        release_inputs();
        release_outputs();
        if (message != nullptr && !PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError, message);
        }
        return nullptr;
    };

    std::size_t total_output_bytes = 0;
    try {
        for (Py_ssize_t tensor_index = 0;
             tensor_index < tensor_count;
             ++tensor_index) {
            PyObject* item = PySequence_Fast_GET_ITEM(
                tensor_sequence.get(), tensor_index);
            int element_size = 0;
            int bit_reorder = 0;
            Py_ssize_t requested_numel = 0;
            PyObject* groups_source = nullptr;
            if (!PyArg_ParseTuple(
                    item,
                    "ipnO",
                    &element_size,
                    &bit_reorder,
                    &requested_numel,
                    &groups_source)) {
                return fail(nullptr);
            }
            if (element_size != 2 && element_size != 4) {
                return fail("element_size must be 2 or 4.");
            }
            if (requested_numel < 0) {
                return fail("numel must be non-negative.");
            }

            const std::size_t numel = static_cast<std::size_t>(
                requested_numel);
            const std::size_t width = static_cast<std::size_t>(element_size);
            if (numel > static_cast<std::size_t>(PY_SSIZE_T_MAX) / width) {
                PyErr_SetString(
                    PyExc_OverflowError,
                    "Decoded tensor byte size exceeds Py_ssize_t.");
                return fail(nullptr);
            }
            const std::size_t raw_bytes = numel * width;
            if (raw_bytes
                > std::numeric_limits<std::size_t>::max()
                    - total_output_bytes) {
                PyErr_SetString(
                    PyExc_OverflowError,
                    "Total decoded tensor size overflowed.");
                return fail(nullptr);
            }
            total_output_bytes += raw_bytes;

            MixedStreamDecodeTensor& request = requests[
                static_cast<std::size_t>(tensor_index)];
            request.element_size = element_size;
            request.bit_reorder = bit_reorder != 0;
            request.numel = numel;
            request.output = PyByteArray_FromStringAndSize(
                nullptr, static_cast<Py_ssize_t>(raw_bytes));
            if (!request.output) {
                return fail(nullptr);
            }
            request.output_data = reinterpret_cast<std::uint8_t*>(
                PyByteArray_AS_STRING(request.output));

            OwnedPyRef group_sequence(PySequence_Fast(
                groups_source, "groups must be a sequence"));
            if (!group_sequence.get()) {
                return fail(nullptr);
            }
            if (PySequence_Fast_GET_SIZE(group_sequence.get()) != element_size) {
                return fail(
                    "groups must contain exactly one group per byte stream.");
            }

            std::array<std::vector<std::size_t>, 4> stream_chunk_indices;
            for (int stream = 0; stream < element_size; ++stream) {
                PyObject* group_source = PySequence_Fast_GET_ITEM(
                    group_sequence.get(), stream);
                OwnedPyRef chunk_sequence(PySequence_Fast(
                    group_source, "each group must be a sequence of chunks"));
                if (!chunk_sequence.get()) {
                    return fail(nullptr);
                }

                std::size_t output_offset = 0;
                const Py_ssize_t chunk_count = PySequence_Fast_GET_SIZE(
                    chunk_sequence.get());
                for (Py_ssize_t chunk_index = 0;
                     chunk_index < chunk_count;
                     ++chunk_index) {
                    PyObject* chunk_source = PySequence_Fast_GET_ITEM(
                        chunk_sequence.get(), chunk_index);
                    const char* kind = nullptr;
                    PyObject* payload_source = nullptr;
                    Py_ssize_t requested_symbols = 0;
                    PyObject* decodebook_source = nullptr;
                    int frame_4x = 0;
                    if (!PyArg_ParseTuple(
                            chunk_source,
                            "sOnOp",
                            &kind,
                            &payload_source,
                            &requested_symbols,
                            &decodebook_source,
                            &frame_4x)) {
                        return fail(nullptr);
                    }
                    if (requested_symbols < 0) {
                        return fail("chunk symbol_count must be non-negative.");
                    }
                    const std::size_t symbol_count = static_cast<std::size_t>(
                        requested_symbols);
                    if (symbol_count > numel - output_offset) {
                        return fail(
                            "chunk symbol counts exceed the tensor numel.");
                    }

                    chunks.emplace_back();
                    MixedStreamDecodeChunk& chunk = chunks.back();
                    chunk.tensor_index = static_cast<std::size_t>(tensor_index);
                    chunk.stream = stream;
                    chunk.output_offset = output_offset;
                    chunk.symbol_count = symbol_count;
                    chunk.frame_4x = frame_4x != 0;
                    if (PyObject_GetBuffer(
                            payload_source,
                            &chunk.payload_view,
                            PyBUF_CONTIG_RO) < 0) {
                        return fail(nullptr);
                    }

                    if (std::strcmp(kind, "raw") == 0) {
                        chunk.mode = MixedStreamDecodeMode::Raw;
                        if (decodebook_source != Py_None) {
                            return fail("raw chunks cannot reference a decodebook.");
                        }
                        if (chunk.payload_view.len != requested_symbols) {
                            return fail(
                                "raw chunk payload size must equal symbol_count.");
                        }
                    } else if (std::strcmp(kind, "zero") == 0) {
                        chunk.mode = MixedStreamDecodeMode::Zero;
                        if (decodebook_source != Py_None) {
                            return fail("zero chunks cannot reference a decodebook.");
                        }
                        if (chunk.payload_view.len != 0) {
                            return fail("zero chunk payload must be empty.");
                        }
                    } else if (std::strcmp(kind, "local_huff0") == 0) {
                        chunk.mode = MixedStreamDecodeMode::LocalHuff0;
                        if (decodebook_source != Py_None) {
                            return fail(
                                "local_huff0 chunks cannot reference a decodebook.");
                        }
                        if (symbol_count == 0 || chunk.payload_view.len <= 0) {
                            return fail(
                                "local_huff0 chunks must contain symbols and payload.");
                        }
                        if (symbol_count > kMaxBlockSize) {
                            return fail(
                                "local_huff0 chunk exceeds the Huff0 block limit.");
                        }
                    } else if (std::strcmp(kind, "shared_huff0") == 0) {
                        chunk.mode = MixedStreamDecodeMode::SharedHuff0;
                        if (decodebook_source == Py_None) {
                            return fail(
                                "shared_huff0 chunks require a decodebook handle.");
                        }
                        if (symbol_count == 0 || chunk.payload_view.len <= 0) {
                            return fail(
                                "shared_huff0 chunks must contain symbols and payload.");
                        }
                        if (symbol_count > kMaxBlockSize) {
                            return fail(
                                "shared_huff0 chunk exceeds the Huff0 block limit.");
                        }
                        chunk.decodebook = decodebook_from_capsule(
                            decodebook_source);
                        Py_INCREF(decodebook_source);
                        chunk.decodebook_capsule = decodebook_source;
                    } else if (std::strcmp(kind, "packed_huff0") == 0) {
                        return fail(
                            "decompress_mixed_streams_many does not support packed_huff0.");
                    } else {
                        return fail("Unsupported mixed-stream decode chunk mode.");
                    }

                    stream_chunk_indices[static_cast<std::size_t>(stream)]
                        .push_back(chunks.size() - 1U);
                    output_offset += symbol_count;
                }
                if (output_offset != numel) {
                    return fail(
                        "Each decoded byte stream must contain exactly numel symbols.");
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
                    for (std::size_t stream = 1; stream < width; ++stream) {
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
                    job.tensor_index = static_cast<std::size_t>(tensor_index);
                    job.output_offset = first.output_offset;
                    job.symbol_count = first.symbol_count;
                    for (std::size_t stream = 0; stream < width; ++stream) {
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
                for (std::size_t stream = 0; stream < width; ++stream) {
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
                    combine_jobs.push_back(
                        {
                            static_cast<std::size_t>(tensor_index),
                            start,
                            std::min(combine_step, numel - start),
                        });
                }
            }
        }
    } catch (const std::bad_alloc&) {
        PyErr_NoMemory();
        return fail(nullptr);
    } catch (const std::exception& error) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError, error.what());
        }
        return fail(nullptr);
    }

    std::string thread_error;
    std::string native_error;
    const std::string* decode_error = nullptr;
    PyThreadState* saved = PyEval_SaveThread();
    try {
        const std::size_t aligned_job_count = aligned_jobs.size();
        const std::size_t first_phase_job_count = aligned_job_count
            + fallback_chunk_jobs.size();
        thread_error = run_parallel_jobs(
            first_phase_job_count,
            total_output_bytes,
            requested_threads,
            [&](std::size_t job_index) {
                if (job_index < aligned_job_count) {
                    MixedStreamAlignedDecodeJob& job = aligned_jobs[job_index];
                    const MixedStreamDecodeTensor& request = requests[
                        job.tensor_index];
                    try {
                        const std::size_t width = static_cast<std::size_t>(
                            request.element_size);
                        std::size_t huff_lane_count = 0;
                        for (std::size_t stream = 0; stream < width; ++stream) {
                            const MixedStreamDecodeMode mode = chunks[
                                job.chunk_indices[stream]].mode;
                            if (mode == MixedStreamDecodeMode::LocalHuff0
                                || mode == MixedStreamDecodeMode::SharedHuff0) {
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
                        scratch.resize(huff_lane_count * job.symbol_count);
                        std::array<const std::uint8_t*, 4> lane_data{};
                        std::size_t scratch_offset = 0;
                        for (std::size_t stream = 0; stream < width; ++stream) {
                            MixedStreamDecodeChunk& chunk = chunks[
                                job.chunk_indices[stream]];
                            const auto* payload = static_cast<
                                const std::uint8_t*>(chunk.payload_view.buf);
                            switch (chunk.mode) {
                            case MixedStreamDecodeMode::Raw:
                                lane_data[stream] = payload;
                                break;
                            case MixedStreamDecodeMode::Zero:
                                lane_data[stream] = nullptr;
                                break;
                            case MixedStreamDecodeMode::LocalHuff0: {
                                std::uint8_t* destination = scratch.data()
                                    + scratch_offset;
                                decompress_local_block_into(
                                    payload,
                                    chunk.payload_view.len,
                                    destination,
                                    chunk.symbol_count);
                                lane_data[stream] = destination;
                                scratch_offset += chunk.symbol_count;
                                break;
                            }
                            case MixedStreamDecodeMode::SharedHuff0: {
                                std::uint8_t* destination = scratch.data()
                                    + scratch_offset;
                                decode_huff0_block_into(
                                    payload,
                                    chunk.payload_view.len,
                                    destination,
                                    chunk.symbol_count,
                                    *chunk.decodebook,
                                    chunk.frame_4x);
                                lane_data[stream] = destination;
                                scratch_offset += chunk.symbol_count;
                                break;
                            }
                            }
                        }
                        restore_mixed_stream_aligned_chunk(
                            request,
                            lane_data,
                            job.output_offset,
                            job.symbol_count);
                    } catch (const std::exception& error) {
                        job.error = error.what();
                    } catch (...) {
                        job.error = "Unknown aligned mixed-stream decode failure.";
                    }
                    return;
                }

                MixedStreamDecodeChunk& chunk = chunks[
                    fallback_chunk_jobs[job_index - aligned_job_count]];
                MixedStreamDecodeTensor& request = requests[chunk.tensor_index];
                std::uint8_t* destination = nullptr;
                if (chunk.symbol_count > 0) {
                    destination = request.lanes[
                        static_cast<std::size_t>(chunk.stream)].data()
                        + chunk.output_offset;
                }
                try {
                    const auto* payload = static_cast<const std::uint8_t*>(
                        chunk.payload_view.buf);
                    switch (chunk.mode) {
                    case MixedStreamDecodeMode::Raw:
                        if (chunk.symbol_count > 0) {
                            std::memcpy(
                                destination, payload, chunk.symbol_count);
                        }
                        break;
                    case MixedStreamDecodeMode::Zero:
                        if (chunk.symbol_count > 0) {
                            std::memset(destination, 0, chunk.symbol_count);
                        }
                        break;
                    case MixedStreamDecodeMode::LocalHuff0:
                        decompress_local_block_into(
                            payload,
                            chunk.payload_view.len,
                            destination,
                            chunk.symbol_count);
                        break;
                    case MixedStreamDecodeMode::SharedHuff0:
                        decode_huff0_block_into(
                            payload,
                            chunk.payload_view.len,
                            destination,
                            chunk.symbol_count,
                            *chunk.decodebook,
                            chunk.frame_4x);
                        break;
                    }
                } catch (const std::exception& error) {
                    chunk.error = error.what();
                } catch (...) {
                    chunk.error = "Unknown mixed-stream decode failure.";
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
                    const MixedStreamCombineJob& job = combine_jobs[job_index];
                    restore_mixed_stream_tensor_range(
                        requests[job.tensor_index], job.start, job.length);
                });
        }
    } catch (const std::exception& error) {
        native_error = error.what();
    } catch (...) {
        native_error = "Unknown model-wide mixed-stream decode failure.";
    }
    PyEval_RestoreThread(saved);

    if (!native_error.empty()) {
        PyErr_SetString(PyExc_RuntimeError, native_error.c_str());
        return fail(nullptr);
    }
    if (!thread_error.empty()) {
        PyErr_SetString(PyExc_RuntimeError, thread_error.c_str());
        return fail(nullptr);
    }
    if (decode_error != nullptr) {
        PyErr_SetString(PyExc_ValueError, decode_error->c_str());
        return fail(nullptr);
    }

    release_inputs();
    PyObject* output = PyList_New(tensor_count);
    if (!output) {
        release_outputs();
        return nullptr;
    }
    for (Py_ssize_t tensor_index = 0;
         tensor_index < tensor_count;
         ++tensor_index) {
        MixedStreamDecodeTensor& request = requests[
            static_cast<std::size_t>(tensor_index)];
        PyList_SET_ITEM(output, tensor_index, request.output);
        request.output = nullptr;
        request.output_data = nullptr;
    }
    return output;
}

PyObject* py_decompress_block(PyObject*, PyObject* args) {
    Py_buffer payload_view{};
    Py_ssize_t symbol_count = 0;
    if (!PyArg_ParseTuple(args, "y*n", &payload_view, &symbol_count)) {
        return nullptr;
    }
    try {
        const auto* payload = static_cast<const std::uint8_t*>(payload_view.buf);
        std::vector<std::uint8_t> decoded = decompress_local_block(
            payload,
            payload_view.len,
            symbol_count);
        PyObject* result = bytes_from_vector(decoded);
        PyBuffer_Release(&payload_view);
        return result;
    } catch (const std::bad_alloc&) {
        PyBuffer_Release(&payload_view);
        PyErr_NoMemory();
        return nullptr;
    } catch (const std::exception& error) {
        PyBuffer_Release(&payload_view);
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

static PyMethodDef methods[] = {
    {"max_block_size", py_max_block_size, METH_NOARGS, "Return Huff0 maximum block size."},
    {"build_codebook", reinterpret_cast<PyCFunction>(py_build_codebook), METH_VARARGS | METH_KEYWORDS, "Build a Huff0 codebook."},
    {"build_codebook_from_counts", reinterpret_cast<PyCFunction>(py_build_codebook_from_counts), METH_VARARGS | METH_KEYWORDS, "Build a Huff0 codebook from a complete 256-bin histogram."},
    {"build_codebook_from_header", py_build_codebook_from_header, METH_VARARGS, "Rebuild a Huff0 encoder table from its serialized header."},
    {"build_decodebook", py_build_decodebook, METH_VARARGS, "Build a Huff0 decode table from a serialized codebook."},
    {"export_code_table", reinterpret_cast<PyCFunction>(py_export_code_table), METH_VARARGS | METH_KEYWORDS, "Export exact Huff0 symbol lengths and canonical code values."},
    {"encode_gpu_bfloat16", reinterpret_cast<PyCFunction>(py_encode_gpu_bfloat16), METH_VARARGS | METH_KEYWORDS, "Encode little-endian BF16 bytes into the NeuralZip GPU layout."},
    {"encode_with_codebook", reinterpret_cast<PyCFunction>(py_encode_with_codebook), METH_VARARGS | METH_KEYWORDS, "Encode using a Huff0 codebook."},
    {"encode_with_codebook_4x", reinterpret_cast<PyCFunction>(py_encode_with_codebook_4x), METH_VARARGS | METH_KEYWORDS, "Encode four Huff0 streams using a precomputed codebook."},
    {"encode_ngram_with_codebook", reinterpret_cast<PyCFunction>(py_encode_ngram_with_codebook), METH_VARARGS | METH_KEYWORDS, "Tokenize n-grams and encode token ids using a Huff0 codebook."},
    {"encode_ngram_with_codebook_4x", reinterpret_cast<PyCFunction>(py_encode_ngram_with_codebook_4x), METH_VARARGS | METH_KEYWORDS, "Tokenize n-grams and encode token ids as four Huff0 streams using a precomputed codebook."},
    {"decompress_with_decodebook", py_decompress_with_decodebook, METH_VARARGS, "Decompress using a Huff0 decodebook."},
    {"decompress_with_decodebook_4x", py_decompress_with_decodebook_4x, METH_VARARGS, "Decompress four Huff0 streams using a precomputed decodebook."},
    {"decompress_ngram_with_decodebook", py_decompress_ngram_with_decodebook, METH_VARARGS, "Decode Huff0 token ids and restore packed n-grams."},
    {"decompress_ngram_with_decodebook_4x", py_decompress_ngram_with_decodebook_4x, METH_VARARGS, "Decode four Huff0 token streams and restore packed n-grams."},
    {"compress_mixed_streams_many", reinterpret_cast<PyCFunction>(py_compress_mixed_streams_many), METH_VARARGS | METH_KEYWORDS, "Encode many tensors with one mixed-stream Huff0 native job queue."},
    {"decompress_mixed_streams_many", reinterpret_cast<PyCFunction>(py_decompress_mixed_streams_many), METH_VARARGS | METH_KEYWORDS, "Decode many mixed-stream tensors with one native job queue."},
    {"decompress_block", py_decompress_block, METH_VARARGS, "Decompress one self-contained Huff0 block."},
    {nullptr, nullptr, 0, nullptr},
};

static PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_huff0_cpp",
    "NeuralZip stream bridge backed by upstream Zstd Huff0.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__huff0_cpp() {
    return PyModule_Create(&module);
}
