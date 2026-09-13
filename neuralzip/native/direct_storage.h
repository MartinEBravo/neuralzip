constexpr std::size_t kSharedChunkMetadataBytes = 25;

struct DirectStorageRequest {
    ReadOnlyBuffer raw;
    unsigned element_size = 0;
    bool bit_reorder = false;
    unsigned mantissa_bits = 0;
    unsigned exponent_bits = 0;
    unsigned stream_id = 0;
    nb::object codebook_owner;
    Huff0Codebook* codebook = nullptr;
    MixedMode mode = MixedMode::Raw;
    std::shared_ptr<const NGramVocabLookup> vocab_lookup;
    int ngram_n = 1;
    Py_ssize_t codebook_id = -1;
    std::size_t chunk_size = 0;
    std::size_t symbol_count = 0;
    std::size_t symbol_offset = 0;
    std::size_t raw_offset = 0;
    std::size_t first_job = 0;
    std::size_t job_count = 0;
};

struct DirectStorageJob {
    std::size_t request_index = 0;
    std::size_t start = 0;
    std::size_t length = 0;
    std::size_t payload_size = 0;
    std::size_t payload_offset = 0;
    std::size_t escape_size = 0;
    std::size_t escape_offset = 0;
    std::size_t tail_size = 0;
    std::size_t tail_offset = 0;
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

nb::tuple compress_direct_storage(
    nb::handle tensor_source,
    nb::handle prefix_source,
    Py_ssize_t min_saving_bytes,
    int requested_threads,
    double compression_threshold) {
    require<nb::value_error>(
        min_saving_bytes >= 0 && requested_threads >= 0,
        "min_saving_bytes and threads must be non-negative.");
    require<nb::value_error>(
        compression_threshold > 0.0 && compression_threshold <= 1.0,
        "compression_threshold must be in (0, 1].");

    nb::tuple tensor_sequence(tensor_source);
    std::vector<DirectStorageRequest> requests(tensor_sequence.size());
    std::vector<DirectStorageJob> jobs;
    std::size_t total_symbols = 0;
    std::size_t total_input_bytes = 0;

    for (std::size_t tensor_index = 0;
         tensor_index < tensor_sequence.size();
         ++tensor_index) {
        nb::tuple item(nb::handle(tensor_sequence[tensor_index].ptr()));
        require<nb::type_error>(
            item.size() == 8,
            "Each direct spec needs data, width, reorder, mantissa/exponent bits, "
            "stream_id, chunk_size, and mode.");
        DirectStorageRequest& request = requests[tensor_index];
        request.raw.acquire(nb::handle(item[0].ptr()));
        request.element_size = nb::cast<unsigned>(nb::handle(item[1].ptr()));
        request.bit_reorder = nb::cast<bool>(nb::handle(item[2].ptr()));
        request.mantissa_bits = nb::cast<unsigned>(nb::handle(item[3].ptr()));
        request.exponent_bits = nb::cast<unsigned>(nb::handle(item[4].ptr()));
        request.stream_id = nb::cast<unsigned>(nb::handle(item[5].ptr()));
        require<nb::value_error>(
            request.element_size == 2 || request.element_size == 4,
            "Direct storage element_size must be 2 or 4.");
        require<nb::value_error>(
            request.mantissa_bits < 8U * request.element_size
                && request.exponent_bits > 0
                && request.exponent_bits < 8U * request.element_size
                && request.mantissa_bits + request.exponent_bits + 1U
                    == 8U * request.element_size,
            "Sign, exponent and mantissa fields must fill the element.");
        require<nb::value_error>(
            request.stream_id < request.element_size, "stream_id is out of range.");
        require<nb::value_error>(
            request.raw.size() >= 0 && request.raw.size() % request.element_size == 0,
            "Direct storage input must contain complete elements.");
        const Py_ssize_t requested_chunk_size =
            nb::cast<Py_ssize_t>(nb::handle(item[6].ptr()));
        require<nb::value_error>(
            requested_chunk_size > 0
                && requested_chunk_size <= static_cast<Py_ssize_t>(kMaxBlockSize),
            "Direct storage chunk_size must be in the Huff0 block range.");
        request.chunk_size = static_cast<std::size_t>(requested_chunk_size);

        nb::tuple mode(nb::handle(item[7].ptr()));
        require<nb::type_error>(
            mode.size() == 5,
            "The direct stream mode must contain kind, codebook, id, "
            "vocabulary, and ngram_n.");
        const std::string kind = nb::cast<std::string>(nb::handle(mode[0].ptr()));
        if (kind == "shared_huff0" || kind == "packed_huff0") {
            request.codebook_owner = nb::borrow<nb::object>(
                nb::handle(mode[1].ptr()));
            request.codebook = capsule_data<Huff0Codebook>(
                nb::handle(mode[1].ptr()), "neuralzip.Huff0Codebook");
            request.codebook_id = nb::cast<Py_ssize_t>(nb::handle(mode[2].ptr()));
            require<nb::value_error>(
                request.codebook_id >= 0,
                "Compressed direct mode requires a non-negative codebook id.");
            if (kind == "packed_huff0") {
                request.mode = MixedMode::PackedHuff0;
                request.ngram_n = nb::cast<int>(nb::handle(mode[4].ptr()));
                require<nb::value_error>(
                    request.ngram_n >= 2 && request.ngram_n <= 8,
                    "packed_huff0 requires ngram_n in [2, 8].");
                request.vocab_lookup = prepare_ngram_vocab_lookup(
                    *request.codebook, nb::handle(mode[3].ptr()), request.ngram_n);
            } else {
                request.mode = MixedMode::SharedHuff0;
            }
        } else {
            require<nb::value_error>(
                kind == "raw",
                "Direct encode supports raw, shared_huff0, or packed_huff0.");
        }

        if (request.mode == MixedMode::PackedHuff0) {
            const std::size_t n = static_cast<std::size_t>(request.ngram_n);
            const std::size_t limit = std::min(
                request.chunk_size, kMaxBlockSize * n);
            request.chunk_size = std::max(n, (limit / n) * n);
        }

        request.symbol_count =
            static_cast<std::size_t>(request.raw.size()) / request.element_size;
        require<std::overflow_error>(
            static_cast<std::size_t>(request.raw.size())
                <= std::numeric_limits<std::size_t>::max() - total_input_bytes,
            "Total direct input size overflowed.");
        request.symbol_offset = total_symbols;
        request.raw_offset = total_input_bytes - total_symbols;
        total_symbols += request.symbol_count;
        total_input_bytes += static_cast<std::size_t>(request.raw.size());
        request.first_job = jobs.size();
        for (std::size_t start = 0;
             start < request.symbol_count;
             start += request.chunk_size) {
            jobs.push_back({
                tensor_index,
                start,
                std::min(request.chunk_size, request.symbol_count - start),
            });
        }
        request.job_count = jobs.size() - request.first_job;
    }

    ReadOnlyBuffer prefix(prefix_source);
    const std::size_t prefix_size = static_cast<std::size_t>(prefix.size());
    require<std::overflow_error>(
        total_input_bytes <= static_cast<std::size_t>(PY_SSIZE_T_MAX) - prefix_size,
        "Direct output exceeds Python's buffer limit.");
    const std::size_t staging_size = prefix_size + total_input_bytes;
    PyObject* raw_output = PyByteArray_FromStringAndSize(
        nullptr, static_cast<Py_ssize_t>(staging_size));
    if (raw_output == nullptr) {
        throw nb::python_error();
    }
    nb::object output = nb::steal<nb::object>(raw_output);
    auto* output_data = reinterpret_cast<std::uint8_t*>(
        PyByteArray_AsString(output.ptr()));
    if (prefix_size > 0) {
        std::memcpy(output_data, prefix.data(), prefix_size);
    }
    const std::size_t raw_base = prefix_size;
    const std::size_t selected_base = prefix_size + total_input_bytes - total_symbols;

    {
        nb::gil_scoped_release release;
        run_parallel_jobs(
            jobs.size(),
            total_input_bytes,
            requested_threads,
            [&](std::size_t job_index) {
                DirectStorageJob& job = jobs[job_index];
                const DirectStorageRequest& request = requests[job.request_index];
                try {
                    std::uint8_t* __restrict selected_destination =
                        output_data + selected_base + request.symbol_offset + job.start;
                    thread_local std::vector<std::uint8_t> symbols;
                    symbols.resize(job.length);
                    const bool aligned = request.bit_reorder && request.exponent_bits == 8
                        && request.stream_id == request.element_size - 1U;
                    const auto extract = request.element_size == 2
                        ? (aligned ? extract_direct_streams<2, true> : extract_direct_streams<2, false>)
                        : (aligned ? extract_direct_streams<4, true> : extract_direct_streams<4, false>);
                    extract(request, job, output_data + raw_base + request.raw_offset, symbols.data());

                    job.payload_size = job.length;
                    if (request.mode == MixedMode::SharedHuff0) {
                        const std::size_t compressed_size = encode_huff0_block_into(
                            symbols.data(),
                            symbols.size(),
                            *request.codebook,
                            selected_destination,
                            job.length,
                            job.length);
                        if (
                            compressed_size > 0
                            && compressed_size
                                + static_cast<std::size_t>(min_saving_bytes)
                                < job.length
                            && static_cast<double>(kSharedChunkMetadataBytes)
                                    + static_cast<double>(compressed_size)
                                < static_cast<double>(job.length)
                                    * compression_threshold) {
                            job.mode = MixedMode::SharedHuff0;
                            job.payload_size = compressed_size;
                        }
                    } else if (request.mode == MixedMode::PackedHuff0) {
                        NGramBlockResult packed = encode_tokenized_ngram_block(
                            tokenize_byte_group_stream_ngrams(
                                symbols.data(),
                                1,
                                false,
                                0,
                                0,
                                static_cast<Py_ssize_t>(job.length),
                                request.ngram_n,
                                *request.vocab_lookup),
                            *request.codebook,
                            min_saving_bytes,
                            packed_output_budget(job.length, compression_threshold));
                        const std::size_t packed_size = packed.payload.size()
                            + packed.escapes.size() + packed.tail.size();
                        if (
                            !packed.payload.empty()
                            && static_cast<double>(kPackedMetadataBytes)
                                    + static_cast<double>(packed_size)
                                < static_cast<double>(job.length)
                                    * compression_threshold) {
                            job.mode = MixedMode::PackedHuff0;
                            job.payload_size = packed.payload.size();
                            job.escape_size = packed.escapes.size();
                            job.tail_size = packed.tail.size();
                            std::uint8_t* destination = selected_destination;
                            std::memcpy(
                                destination,
                                packed.payload.data(),
                                packed.payload.size());
                            destination += packed.payload.size();
                            if (!packed.escapes.empty()) {
                                std::memcpy(
                                    destination,
                                    packed.escapes.data(),
                                    packed.escapes.size());
                            }
                            destination += packed.escapes.size();
                            if (!packed.tail.empty()) {
                                std::memcpy(
                                    destination,
                                    packed.tail.data(),
                                    packed.tail.size());
                            }
                        }
                    }
                    if (job.mode == MixedMode::Raw && job.length > 0) {
                        std::memcpy(
                            selected_destination, symbols.data(), job.length);
                    }
                } catch (const std::exception& error) {
                    job.error = error.what();
                } catch (...) {
                    job.error = "Unknown direct encoding failure.";
                }
            });
    }

    for (const auto& job : jobs) {
        if (!job.error.empty()) {
            throw nb::value_error(job.error.c_str());
        }
    }

    std::size_t cursor = selected_base;
    for (auto& job : jobs) {
        const DirectStorageRequest& request = requests[job.request_index];
        const std::size_t source_offset = selected_base
            + request.symbol_offset + job.start;
        job.payload_offset = cursor;
        job.escape_offset = cursor + job.payload_size;
        job.tail_offset = job.escape_offset + job.escape_size;
        const std::size_t body_size =
            job.payload_size + job.escape_size + job.tail_size;
        std::memmove(
            output_data + cursor,
            output_data + source_offset,
            body_size);
        cursor += body_size;
    }
    if (PyByteArray_Resize(output.ptr(), static_cast<Py_ssize_t>(cursor)) < 0) {
        throw nb::python_error();
    }

    nb::list tensor_groups;
    for (const auto& request : requests) {
        nb::list groups;
        for (unsigned lane = 0; lane < request.element_size; ++lane) {
            nb::list chunks;
            for (std::size_t index = 0; index < request.job_count; ++index) {
                const DirectStorageJob& job = jobs[request.first_job + index];
                if (lane == request.stream_id) {
                    chunks.append(nb::make_tuple(
                        mixed_mode_name(job.mode), job.length,
                        job.mode == MixedMode::Raw ? -1 : request.codebook_id,
                        job.payload_offset, job.payload_size,
                        job.escape_offset, job.escape_size, job.tail_offset, job.tail_size));
                } else {
                    chunks.append(nb::make_tuple(
                        "raw", job.length, -1,
                        raw_base + request.raw_offset + job.start
                            + (lane - (lane > request.stream_id)) * request.symbol_count,
                        job.length, 0, 0, 0, 0));
                }
            }
            groups.append(std::move(chunks));
        }
        tensor_groups.append(nb::tuple(groups));
    }
    return nb::make_tuple(std::move(output), std::move(tensor_groups));
}
