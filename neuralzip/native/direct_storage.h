constexpr std::size_t kSharedChunkMetadataBytes = 25;

struct DirectBf16Request {
    ReadOnlyBuffer raw;
    nb::object codebook_owner;
    Huff0Codebook* codebook = nullptr;
    MixedMode mode = MixedMode::Raw;
    NGramVocabLookup vocab_lookup;
    int ngram_n = 1;
    Py_ssize_t codebook_id = -1;
    std::size_t chunk_size = 0;
    std::size_t symbol_count = 0;
    std::size_t symbol_offset = 0;
    std::size_t first_job = 0;
    std::size_t job_count = 0;
};

struct DirectBf16Job {
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

nb::tuple compress_bf16_storage(
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
    std::vector<DirectBf16Request> requests(tensor_sequence.size());
    std::vector<DirectBf16Job> jobs;
    std::size_t total_symbols = 0;

    for (std::size_t tensor_index = 0;
         tensor_index < tensor_sequence.size();
         ++tensor_index) {
        nb::tuple item(nb::handle(tensor_sequence[tensor_index].ptr()));
        require<nb::type_error>(
            item.size() == 3,
            "Each BF16 spec must contain data, chunk_size, and exponent mode.");
        DirectBf16Request& request = requests[tensor_index];
        request.raw.acquire(nb::handle(item[0].ptr()));
        require<nb::value_error>(
            request.raw.size() >= 0 && request.raw.size() % 2 == 0,
            "BF16 input must contain complete two-byte elements.");
        const Py_ssize_t requested_chunk_size =
            nb::cast<Py_ssize_t>(nb::handle(item[1].ptr()));
        require<nb::value_error>(
            requested_chunk_size > 0
                && requested_chunk_size <= static_cast<Py_ssize_t>(kMaxBlockSize),
            "BF16 chunk_size must be in the Huff0 block range.");
        request.chunk_size = static_cast<std::size_t>(requested_chunk_size);

        nb::tuple mode(nb::handle(item[2].ptr()));
        require<nb::type_error>(
            mode.size() == 5,
            "The BF16 exponent mode must contain kind, codebook, id, "
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
                "Compressed BF16 mode requires a non-negative codebook id.");
            if (kind == "packed_huff0") {
                request.mode = MixedMode::PackedHuff0;
                request.ngram_n = nb::cast<int>(nb::handle(mode[4].ptr()));
                require<nb::value_error>(
                    request.ngram_n >= 2 && request.ngram_n <= 8,
                    "packed_huff0 requires ngram_n in [2, 8].");
                ReadOnlyBuffer vocab(nb::handle(mode[3].ptr()));
                request.vocab_lookup = build_ngram_vocab_lookup(vocab_from_bytes(
                    vocab.data(), vocab.size(), request.ngram_n));
            } else {
                request.mode = MixedMode::SharedHuff0;
            }
        } else {
            require<nb::value_error>(
                kind == "raw",
                "Direct BF16 encode supports raw, shared_huff0, or packed_huff0.");
        }

        if (request.mode == MixedMode::PackedHuff0) {
            const std::size_t n = static_cast<std::size_t>(request.ngram_n);
            const std::size_t limit = std::min(
                request.chunk_size, kMaxBlockSize * n);
            request.chunk_size = std::max(n, (limit / n) * n);
        }

        request.symbol_count =
            static_cast<std::size_t>(request.raw.size()) / 2U;
        require<std::overflow_error>(
            request.symbol_count
                <= std::numeric_limits<std::size_t>::max() - total_symbols,
            "Total BF16 symbol count overflowed.");
        request.symbol_offset = total_symbols;
        total_symbols += request.symbol_count;
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
        total_symbols
            <= (static_cast<std::size_t>(PY_SSIZE_T_MAX) - prefix_size) / 2U,
        "Direct BF16 output exceeds Python's buffer limit.");
    const std::size_t staging_size = prefix_size + 2U * total_symbols;
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
    const std::size_t exponent_base = prefix_size + total_symbols;

    {
        nb::gil_scoped_release release;
        run_parallel_jobs(
            jobs.size(),
            total_symbols * 2U,
            requested_threads,
            [&](std::size_t job_index) {
                DirectBf16Job& job = jobs[job_index];
                const DirectBf16Request& request = requests[job.request_index];
                try {
                    const std::uint8_t* __restrict source =
                        request.raw.data() + job.start * 2U;
                    std::uint8_t* __restrict raw_destination =
                        output_data + raw_base + request.symbol_offset + job.start;
                    std::uint8_t* __restrict exponent_destination =
                        output_data + exponent_base + request.symbol_offset + job.start;
                    thread_local std::vector<std::uint8_t> exponents;
                    exponents.resize(job.length);
                    std::uint8_t* __restrict exponent_values =
                        exponents.data();
                    for (std::size_t index = 0; index < job.length; ++index) {
                        const std::uint8_t low = source[index * 2U];
                        const std::uint8_t high = source[index * 2U + 1U];
                        raw_destination[index] = static_cast<std::uint8_t>(
                            (low & 0x7FU) | (high & 0x80U));
                        exponent_values[index] = static_cast<std::uint8_t>(
                            ((high << 1U) & 0xFEU) | (low >> 7U));
                    }

                    job.payload_size = job.length;
                    if (request.mode == MixedMode::SharedHuff0) {
                        const std::size_t compressed_size = encode_huff0_block_into(
                            exponent_values,
                            exponents.size(),
                            *request.codebook,
                            exponent_destination,
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
                                exponent_values,
                                1,
                                false,
                                0,
                                0,
                                static_cast<Py_ssize_t>(job.length),
                                request.ngram_n,
                                request.vocab_lookup),
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
                            std::uint8_t* destination = exponent_destination;
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
                            exponent_destination, exponent_values, job.length);
                    }
                } catch (const std::exception& error) {
                    job.error = error.what();
                } catch (...) {
                    job.error = "Unknown direct BF16 encoding failure.";
                }
            });
    }

    for (const auto& job : jobs) {
        if (!job.error.empty()) {
            throw nb::value_error(job.error.c_str());
        }
    }

    std::size_t cursor = exponent_base;
    for (auto& job : jobs) {
        const DirectBf16Request& request = requests[job.request_index];
        const std::size_t source_offset = exponent_base
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
        nb::list raw_chunks;
        nb::list exponent_chunks;
        for (std::size_t index = 0; index < request.job_count; ++index) {
            const DirectBf16Job& job = jobs[request.first_job + index];
            raw_chunks.append(nb::make_tuple(
                "raw",
                job.length,
                -1,
                raw_base + request.symbol_offset + job.start,
                job.length,
                0,
                0,
                0,
                0));
            exponent_chunks.append(nb::make_tuple(
                mixed_mode_name(job.mode),
                job.length,
                job.mode == MixedMode::Raw ? -1 : request.codebook_id,
                job.payload_offset,
                job.payload_size,
                job.escape_offset,
                job.escape_size,
                job.tail_offset,
                job.tail_size));
        }
        tensor_groups.append(nb::make_tuple(
            std::move(raw_chunks), std::move(exponent_chunks)));
    }
    return nb::make_tuple(std::move(output), std::move(tensor_groups));
}
