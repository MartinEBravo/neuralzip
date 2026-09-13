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
    double compression_threshold) {
    const double compressed_bytes = static_cast<double>(kPackedMetadataBytes)
        + static_cast<double>(packed.payload.size())
        + static_cast<double>(packed.escapes.size())
        + static_cast<double>(packed.tail.size());
    if (packed.payload.empty()
        || compressed_bytes
            >= static_cast<double>(raw_size) * compression_threshold) {
        return false;
    }
    result.mode = MixedMode::PackedHuff0;
    result.payload = std::move(packed.payload);
    result.escapes = std::move(packed.escapes);
    result.tail = std::move(packed.tail);
    return true;
}

void process_mixed_block(
    MixedBlockResult& result,
    const MixedStreamSpec& spec,
    Py_ssize_t min_saving_bytes,
    double compression_threshold) {
    result.mode = MixedMode::Raw;
    if (spec.mode == MixedMode::Raw) {
        return;
    }
    require(
        spec.codebook != nullptr,
        "Mixed shared mode is missing a codebook.");
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
        }
    }
    // A shared post-setup mode must never build a target-local tree. Raw is
    // already the exact fallback when the precomputed table is not worthwhile.
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
    require(
        spec.codebook != nullptr,
        "Mixed packed mode is missing a codebook.");
    NGramBlockResult packed = encode_tokenized_ngram_block(
        tokenize_byte_group_stream_ngrams(
            raw,
            element_size,
            bit_reorder,
            stream,
            start,
            static_cast<Py_ssize_t>(length),
            spec.ngram_n,
            *spec.vocab_lookup),
        *spec.codebook,
        min_saving_bytes,
        packed_output_budget(length, compression_threshold));
    if (finish_packed_block(
            result,
            packed,
            length,
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

template <typename T>
void delete_capsule(void* pointer) noexcept {
    delete static_cast<T*>(pointer);
}

template <typename T>
T* capsule_data(nb::handle source, const char* name) {
    nb::capsule capsule = nb::borrow<nb::capsule>(source);
    return static_cast<T*>(capsule.data(name));
}

std::vector<MixedStreamSpec> parse_mixed_stream_specs(
    nb::handle mode_source,
    int element_size,
    std::size_t base_chunk_size) {
    nb::tuple mode_sequence(mode_source);
    require<nb::value_error>(
        mode_sequence.size() == static_cast<std::size_t>(element_size),
        "stream_modes must contain exactly one mode per byte stream.");

    std::vector<MixedStreamSpec> specs(
        static_cast<std::size_t>(element_size));
    for (int stream = 0; stream < element_size; ++stream) {
        nb::tuple item(nb::handle(mode_sequence[stream].ptr()));
        require<nb::type_error>(
            item.size() == 4,
            "Each stream mode must contain kind, codebook, vocabulary, and ngram_n.");

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
            spec.codebook = capsule_data<Huff0Codebook>(
                codebook_source, "neuralzip.Huff0Codebook");
            spec.ngram_n = 1;
        } else if (kind == "packed_huff0") {
            require<nb::value_error>(
                ngram_n >= 2 && ngram_n <= 8,
                "packed_huff0 requires ngram_n in [2, 8].");
            spec.mode = MixedMode::PackedHuff0;
            spec.codebook_owner = nb::borrow<nb::object>(codebook_source);
            spec.codebook = capsule_data<Huff0Codebook>(
                codebook_source, "neuralzip.Huff0Codebook");
            spec.vocab_lookup = prepare_ngram_vocab_lookup(
                *spec.codebook, vocab_source, ngram_n);
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
            chunks.append(nb::make_tuple(
                mixed_mode_name(result.mode),
                bytes_from_vector(body),
                result.symbol_count,
                bytes_from_vector(result.escapes),
                bytes_from_vector(result.tail)));
        }
        groups.append(std::move(chunks));
    }
    return groups;
}
