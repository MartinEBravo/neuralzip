class ReadOnlyBuffer {
public:
    ReadOnlyBuffer() = default;

    explicit ReadOnlyBuffer(nb::handle source) { acquire(source); }

    ~ReadOnlyBuffer() { release(); }

    ReadOnlyBuffer(const ReadOnlyBuffer&) = delete;
    ReadOnlyBuffer& operator=(const ReadOnlyBuffer&) = delete;

    ReadOnlyBuffer(ReadOnlyBuffer&& other) noexcept : view_(other.view_) {
        other.view_ = {};
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

    Py_ssize_t size() const { return view_.len; }

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

template <typename Error = std::runtime_error>
void require(bool condition, const char* message) { if (!condition) throw Error(message); }

template <nb::builtin_exception (*Error)(const char*)>
void require(bool condition, const char* message) { if (!condition) throw Error(message); }

struct MixedHuffmanTable {
    struct Node { std::array<int, 2> children{-1, -1}; int symbol = -1; };
    std::vector<std::uint8_t> lengths;
    std::vector<std::uint64_t> codes;
    std::vector<std::uint16_t> vocabulary;
    std::vector<std::uint16_t> pair_tokens;
    std::vector<Node> nodes;
    std::array<std::uint32_t, 4096> fast{};
};

struct Huff0Codebook {
    std::vector<HUF_CElt> ctable;
    std::vector<std::uint8_t> serialized;
    bool covers_all_symbols = false;
    std::shared_ptr<const MixedHuffmanTable> mixed;
};

struct Huff0Decodebook {
    std::vector<HUF_DTable> dtable_x1;
    std::vector<HUF_DTable> dtable_x2;
    std::shared_ptr<const MixedHuffmanTable> mixed;
};

enum class MixedMode {
    Raw,
    SharedHuff0,
    MixedHuffman,
};

struct MixedStreamDecodeChunk {
    MixedMode mode = MixedMode::Raw;
    ReadOnlyBuffer payload;
    nb::object decodebook_owner;
    Huff0Decodebook* decodebook = nullptr;
    std::size_t output_offset = 0;
    std::size_t symbol_count = 0;
};

template <typename T>
void delete_capsule(void* pointer) noexcept { delete static_cast<T*>(pointer); }

template <typename T>
T* capsule_data(nb::handle source, const char* name) {
    return static_cast<T*>(nb::borrow<nb::capsule>(source).data(name));
}

const char* mixed_mode_name(MixedMode mode) {
    static constexpr const char* names[] = {"raw", "shared_huff0", "mixed_huffman"};
    return names[static_cast<unsigned>(mode)];
}

struct MixedStreamDecodeTensor {
    int element_size = 0;
    bool bit_reorder = false;
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


template <typename Worker>
void run_parallel_jobs(
    std::size_t job_count,
    std::size_t total_input_bytes,
    int requested_threads,
    Worker&& process_job) {
    if (job_count == 0) {
        return;
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
        return;
    }

    std::exception_ptr launch_error;
    std::vector<std::thread> workers;
    try {
        workers.reserve(static_cast<std::size_t>(worker_count));
        for (int index = 0; index < worker_count; ++index) {
            workers.emplace_back(worker);
        }
    } catch (...) {
        launch_error = std::current_exception();
    }
    for (auto& thread : workers) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    if (launch_error) {
        std::rethrow_exception(launch_error);
    }
}

void check_huf(size_t code) {
    if (HUF_isError(code)) {
        throw std::runtime_error(HUF_getErrorName(code));
    }
}

nb::bytes bytes_from_vector(const std::vector<std::uint8_t>& data) {
    return nb::bytes(data.data(), data.size());
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
