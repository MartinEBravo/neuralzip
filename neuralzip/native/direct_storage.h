constexpr std::size_t kSharedChunkMetadataBytes = 25;

struct DirectStorageMode {
    nb::object owner;
    Huff0Codebook* codebook = nullptr;
    MixedMode kind = MixedMode::Raw;
    Py_ssize_t id = -1;
};

struct DirectStorageRequest {
    ReadOnlyBuffer raw;
    unsigned element_size = 0;
    bool bit_reorder = false;
    unsigned mantissa_bits = 0;
    unsigned exponent_bits = 0;
    unsigned stream_id = 0;
    std::vector<DirectStorageMode> modes;
    std::size_t chunk_size = 0, symbol_count = 0;
    std::size_t symbol_offset = 0, raw_offset = 0, first_job = 0, job_count = 0;
};

struct DirectStorageJob {
    std::size_t request_index = 0, start = 0, length = 0;
    std::size_t payload_size = 0, payload_offset = 0;
    MixedMode mode = MixedMode::Raw;
    std::string error;
};

template <unsigned ElementSize, bool ByteAlignedExponent>
void extract_direct_streams(
    const DirectStorageRequest& request,
    const DirectStorageJob& job,
    std::uint8_t* raw_destination,
    std::uint8_t* symbols) {
    const unsigned stream = ByteAlignedExponent ? ElementSize - 1U : request.stream_id;
    const unsigned mantissa = ByteAlignedExponent ? ElementSize * 8U - 9U : request.mantissa_bits;
    const unsigned exponent = ByteAlignedExponent ? 8U : request.exponent_bits;
    const bool reorder = ByteAlignedExponent || request.bit_reorder;
    const std::uint32_t mantissa_mask = (1U << mantissa) - 1U;
    const std::uint32_t sign_mask = 1U << mantissa;
    const std::uint32_t exponent_mask = ((1U << exponent) - 1U) << (mantissa + 1U);
    std::array<std::uint8_t*, ElementSize> lanes{};
    for (unsigned lane = 0; lane < ElementSize; ++lane) {
        if (lane != stream) {
            lanes[lane] = raw_destination + job.start
                + (lane - (lane > stream)) * request.symbol_count;
        }
    }
    const std::uint8_t* source = request.raw.data() + job.start * ElementSize;
    for (std::size_t index = 0; index < job.length; ++index) {
        std::uint32_t value = 0;
        for (unsigned lane = 0; lane < ElementSize; ++lane) {
            value |= static_cast<std::uint32_t>(source[index * ElementSize + lane])
                << (8U * lane);
        }
        if (reorder) {
            value = (value & mantissa_mask) | ((value >> exponent) & sign_mask)
                | ((value << 1U) & exponent_mask);
        }
        for (unsigned lane = 0; lane < ElementSize; ++lane) {
            const auto byte = static_cast<std::uint8_t>(value >> (8U * lane));
            if (lane == stream) {
                symbols[index] = byte;
            } else {
                lanes[lane][index] = byte;
            }
        }
    }
}


nb::tuple compress_direct_storage(nb::handle tensor_source, nb::handle prefix_source,
    Py_ssize_t min_saving_bytes, int requested_threads, double compression_threshold) {
    require<nb::value_error>(min_saving_bytes >= 0 && requested_threads >= 0,
        "min_saving_bytes and threads must be non-negative.");
    require<nb::value_error>(compression_threshold > 0 && compression_threshold <= 1,
        "compression_threshold must be in (0, 1].");
    nb::tuple tensor_sequence(tensor_source);
    std::vector<DirectStorageRequest> requests(tensor_sequence.size());
    std::vector<DirectStorageJob> jobs;
    std::size_t total_symbols = 0, total_input_bytes = 0;
    for (std::size_t i = 0; i < requests.size(); ++i) {
        nb::tuple item(nb::handle(tensor_sequence[i].ptr()));
        require<nb::type_error>(item.size() == 8, "Direct tensor spec must have eight fields.");
        auto& request = requests[i];
        request.raw.acquire(nb::handle(item[0].ptr()));
        request.element_size = nb::cast<unsigned>(nb::handle(item[1].ptr()));
        request.bit_reorder = nb::cast<bool>(nb::handle(item[2].ptr()));
        request.mantissa_bits = nb::cast<unsigned>(nb::handle(item[3].ptr()));
        request.exponent_bits = nb::cast<unsigned>(nb::handle(item[4].ptr()));
        request.stream_id = nb::cast<unsigned>(nb::handle(item[5].ptr()));
        const auto chunk_size = nb::cast<Py_ssize_t>(nb::handle(item[6].ptr()));
        require<nb::value_error>(request.element_size == 2 || request.element_size == 4,
            "Direct storage element_size must be 2 or 4.");
        require<nb::value_error>(request.mantissa_bits < 8U * request.element_size
            && request.exponent_bits > 0 && request.exponent_bits < 8U * request.element_size
            && request.mantissa_bits + request.exponent_bits + 1U == 8U * request.element_size,
            "Sign, exponent and mantissa fields must fill the element.");
        require<nb::value_error>(request.stream_id < request.element_size && chunk_size > 0,
            "Invalid selected stream or chunk size.");
        require<nb::value_error>(request.raw.size() % request.element_size == 0,
            "Direct storage input must contain complete elements.");
        request.chunk_size = static_cast<std::size_t>(chunk_size);
        request.symbol_count = static_cast<std::size_t>(request.raw.size()) / request.element_size;
        const auto raw_size = static_cast<std::size_t>(request.raw.size());
        require<std::overflow_error>(raw_size <= std::numeric_limits<std::size_t>::max() - total_input_bytes,
            "Total direct input size overflowed.");
        request.symbol_offset = total_symbols;
        request.raw_offset = total_input_bytes - total_symbols;
        total_symbols += request.symbol_count;
        total_input_bytes += raw_size;
        request.first_job = jobs.size();
        for (std::size_t start = 0; start < request.symbol_count; start += request.chunk_size)
            jobs.push_back({i, start, std::min(request.chunk_size, request.symbol_count - start)});
        request.job_count = jobs.size() - request.first_job;
        nb::tuple modes(nb::handle(item[7].ptr()));
        require<nb::value_error>(modes.size() == request.job_count,
            "Each source chunk needs exactly one frozen mode.");
        request.modes.resize(modes.size());
        for (std::size_t j = 0; j < modes.size(); ++j) {
            nb::tuple source(nb::handle(modes[j].ptr()));
            require<nb::type_error>(source.size() == 3, "Chunk mode needs kind, table, and id.");
            auto& mode = request.modes[j];
            const auto kind = nb::cast<std::string>(nb::handle(source[0].ptr()));
            if (kind == "raw") continue;
            require<nb::value_error>(kind == "shared_huff0" || kind == "mixed_huffman",
                "Unsupported chunk encoding mode.");
            mode.owner = nb::borrow<nb::object>(nb::handle(source[1].ptr()));
            mode.codebook = capsule_data<Huff0Codebook>(nb::handle(source[1].ptr()), "neuralzip.Huff0Codebook");
            mode.id = nb::cast<Py_ssize_t>(nb::handle(source[2].ptr()));
            require<nb::value_error>(mode.id >= 0, "Compressed chunks need a codebook id.");
            mode.kind = kind == "mixed_huffman" ? MixedMode::MixedHuffman : MixedMode::SharedHuff0;
            require<nb::value_error>(static_cast<bool>(mode.codebook->mixed) == (kind == "mixed_huffman"),
                "Codebook format does not match chunk mode.");
        }
    }
    ReadOnlyBuffer prefix(prefix_source);
    const auto prefix_size = static_cast<std::size_t>(prefix.size());
    require<std::overflow_error>(prefix_size <= static_cast<std::size_t>(PY_SSIZE_T_MAX)
        && total_input_bytes <= static_cast<std::size_t>(PY_SSIZE_T_MAX) - prefix_size,
        "Direct output exceeds Python's buffer limit.");
    nb::bytearray output(static_cast<const void*>(nullptr), prefix_size + total_input_bytes);
    auto* output_data = static_cast<std::uint8_t*>(output.data());
    if (prefix_size) std::memcpy(output_data, prefix.data(), prefix_size);
    const std::size_t raw_base = prefix_size;
    const std::size_t selected_base = prefix_size + total_input_bytes - total_symbols;
    {
        nb::gil_scoped_release release;
        run_parallel_jobs(jobs.size(), total_input_bytes, requested_threads, [&](std::size_t j) {
            auto& job = jobs[j];
            const auto& request = requests[job.request_index];
            const auto& mode = request.modes[j - request.first_job];
            try {
                auto* destination = output_data + selected_base + request.symbol_offset + job.start;
                thread_local std::vector<std::uint8_t> symbols;
                symbols.resize(job.length);
                const bool aligned = request.bit_reorder && request.exponent_bits == 8
                    && request.stream_id == request.element_size - 1U;
                const auto extract = request.element_size == 2
                    ? (aligned ? extract_direct_streams<2, true> : extract_direct_streams<2, false>)
                    : (aligned ? extract_direct_streams<4, true> : extract_direct_streams<4, false>);
                extract(request, job, output_data + raw_base + request.raw_offset, symbols.data());
                job.payload_size = job.length;
                const auto budget = static_cast<Py_ssize_t>(job.length * compression_threshold)
                    - static_cast<Py_ssize_t>(kSharedChunkMetadataBytes);
                std::size_t size = 0;
                if (budget > 0 && mode.kind == MixedMode::SharedHuff0)
                    size = encode_huff0_block_into(symbols.data(), symbols.size(), *mode.codebook,
                        destination, job.length, static_cast<std::size_t>(budget));
                else if (budget > 0 && mode.kind == MixedMode::MixedHuffman)
                    size = encode_mixed_block_into(symbols.data(), symbols.size(), *mode.codebook->mixed,
                        destination, static_cast<std::size_t>(budget));
                if (size && size + static_cast<std::size_t>(min_saving_bytes) < job.length
                    && size + kSharedChunkMetadataBytes < job.length * compression_threshold) {
                    job.mode = mode.kind;
                    job.payload_size = size;
                } else if (job.length) std::memcpy(destination, symbols.data(), job.length);
            } catch (const std::exception& error) { job.error = error.what(); }
            catch (...) { job.error = "Unknown native encoding failure."; }
        });
    }
    for (const auto& job : jobs)
        if (!job.error.empty()) throw nb::value_error(job.error.c_str());
    std::size_t cursor = selected_base;
    for (auto& job : jobs) {
        const auto& request = requests[job.request_index];
        const auto source = selected_base + request.symbol_offset + job.start;
        job.payload_offset = cursor;
        std::memmove(output_data + cursor, output_data + source, job.payload_size);
        cursor += job.payload_size;
    }
    if (PyByteArray_Resize(output.ptr(), static_cast<Py_ssize_t>(cursor)) < 0) throw nb::python_error();
    nb::list tensor_groups;
    for (const auto& request : requests) {
        nb::list groups;
        for (unsigned lane = 0; lane < request.element_size; ++lane) {
            nb::list chunks;
            for (std::size_t j = 0; j < request.job_count; ++j) {
                const auto& job = jobs[request.first_job + j];
                if (lane == request.stream_id)
                    chunks.append(nb::make_tuple(mixed_mode_name(job.mode), job.length,
                        job.mode == MixedMode::Raw ? -1 : request.modes[j].id,
                        job.payload_offset, job.payload_size));
                else
                    chunks.append(nb::make_tuple("raw", job.length, -1,
                        raw_base + request.raw_offset + job.start
                            + (lane - (lane > request.stream_id)) * request.symbol_count, job.length));
            }
            groups.append(std::move(chunks));
        }
        tensor_groups.append(nb::tuple(groups));
    }
    return nb::make_tuple(std::move(output), std::move(tensor_groups));
}
