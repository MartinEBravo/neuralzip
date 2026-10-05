nb::tuple codebook_to_python(std::unique_ptr<Huff0Codebook> codebook) {
    Huff0Codebook* native = codebook.get();
    nb::capsule handle(native, "neuralzip.Huff0Codebook", delete_capsule<Huff0Codebook>);
    codebook.release();
    return nb::make_tuple(
        std::move(handle), bytes_from_vector(native->serialized));
}

std::array<unsigned, 256> parse_symbol_counts(
    nb::handle source,
    unsigned prior_count) {
    nb::tuple sequence(source);
    require<nb::value_error>(
        sequence.size() == 256,
        "symbol_counts must contain exactly 256 entries.");

    std::array<std::uint64_t, 256> counts{};
    std::uint64_t total = 0;
    for (std::size_t index = 0; index < counts.size(); ++index) {
        const auto count = PyLong_AsUnsignedLongLong(
            sequence[index].ptr());
        if (PyErr_Occurred()) {
            throw nb::python_error();
        }
        require<std::overflow_error>(
            total <= std::numeric_limits<std::uint64_t>::max() - count,
            "Huff0 symbol count overflow.");
        counts[index] = count;
        total += count;
    }

    const std::uint64_t prior_total =
        static_cast<std::uint64_t>(prior_count) * counts.size();
    require<nb::value_error>(
        prior_total < kMaxHistogramTotal,
        "prior_count is too large for a 256-symbol Huff0 table.");
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
        require<std::overflow_error>(
            shift < 64,
            "Huff0 symbol count overflow.");
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

nb::tuple build_codebook_from_counts(
    nb::handle count_source,
    Py_ssize_t prior_count,
    Py_ssize_t max_bits) {
    require<nb::value_error>(
        prior_count >= 0 && prior_count <= static_cast<Py_ssize_t>(
            std::numeric_limits<unsigned>::max()),
        "prior_count must fit in unsigned.");
    require<nb::value_error>(
        max_bits >= 1 && max_bits <= static_cast<Py_ssize_t>(kMaxTableLog),
        "max_bits must be in [1, 12].");
    std::array<unsigned, 256> counts = parse_symbol_counts(
        count_source, static_cast<unsigned>(prior_count));
    return codebook_to_python(make_codebook_from_counts(
        counts, static_cast<unsigned>(max_bits)));
}

nb::tuple build_codebook_from_header(nb::handle header_source, nb::handle vocab_source) {
    ReadOnlyBuffer header(header_source);
    ReadOnlyBuffer vocab(vocab_source);
    return codebook_to_python(make_codebook_from_header(
        header.data(), header.size(), vocab.data(), vocab.size()));
}

nb::capsule build_decodebook(nb::handle header_source, nb::handle vocab_source) {
    ReadOnlyBuffer header(header_source);
    ReadOnlyBuffer vocab(vocab_source);
    auto decodebook = make_decodebook_from_header(
        header.data(), header.size(), vocab.data(), vocab.size());
    Huff0Decodebook* native = decodebook.get();
    nb::capsule handle(native, "neuralzip.Huff0Decodebook", delete_capsule<Huff0Decodebook>);
    decodebook.release();
    return handle;
}

nb::tuple export_code_table(
    nb::handle header_source,
    Py_ssize_t max_bits,
    nb::handle vocab_source) {
    require<nb::value_error>(
        max_bits > 0 && max_bits <= kGPUCodeMaxBits,
        "max_bits must be in [1, 56].");
    ReadOnlyBuffer header(header_source);
    ReadOnlyBuffer vocab(vocab_source);
    const auto table = gpu_code_table(header.data(), header.size(), vocab.data(), vocab.size());
    const unsigned longest = *std::max_element(table->lengths.begin(), table->lengths.end());
    require(longest <= max_bits, "GPU codebook exceeds the requested max_bits limit.");
    nb::list values;
    for (const auto value : table->codes) {
        values.append(value);
    }
    return nb::make_tuple(
        nb::bytes(table->lengths.data(), table->lengths.size()),
        nb::tuple(values),
        longest);
}

nb::tuple encode_gpu_tensor(
    nb::handle raw_source,
    nb::handle header_source,
    unsigned element_size,
    bool bit_reorder,
    unsigned mantissa_bits,
    unsigned exponent_bits,
    unsigned stream_id,
    Py_ssize_t bytes_per_thread,
    Py_ssize_t threads_per_block,
    nb::handle vocab_source,
    nb::handle codebook_handle) {
    require<nb::value_error>(
        bytes_per_thread == 8 && threads_per_block == 512,
        "The GPU ABI requires 8 bytes per thread and 512 threads per block.");
    ReadOnlyBuffer raw(raw_source);
    ReadOnlyBuffer header(header_source);
    ReadOnlyBuffer vocab(vocab_source);
    require<nb::value_error>(
        element_size == 2 || element_size == 4,
        "GPU element_size must be 2 or 4.");
    require<nb::value_error>(
        mantissa_bits < 8U * element_size && exponent_bits > 0
            && exponent_bits < 8U * element_size
            && mantissa_bits + exponent_bits + 1U == 8U * element_size,
        "Sign, exponent and mantissa fields must fill the element.");
    require<nb::value_error>(stream_id < element_size, "GPU stream_id is out of range.");
    require(raw.size() % element_size == 0, "Input must contain complete elements.");
    const std::size_t numel = static_cast<std::size_t>(raw.size()) / element_size;
    require<std::overflow_error>(
        numel <= kGPUDecoderIndexLimit,
        "Element count exceeds the int32 CUDA decoder ABI limit.");

    GPULayoutEncoding encoded;
    std::shared_ptr<const MixedHuffmanTable> table;
    if (codebook_handle.is_none()) {
        table = gpu_code_table(header.data(), header.size(), vocab.data(), vocab.size());
    } else {
        auto* book = capsule_data<Huff0Codebook>(codebook_handle, "neuralzip.Huff0Codebook");
        if (!book->gpu) book->gpu = book->mixed ? book->mixed
            : gpu_code_table(header.data(), header.size(), vocab.data(), vocab.size());
        table = book->gpu;
    }
    {
        nb::gil_scoped_release release;
        // Select constant layout parameters outside the symbol loops.
        const bool aligned_exponent = bit_reorder && exponent_bits == 8
            && stream_id == element_size - 1U;
        const auto encode = element_size == 2
            ? (aligned_exponent ? encode_gpu_layout<2, true> : encode_gpu_layout<2, false>)
            : (aligned_exponent ? encode_gpu_layout<4, true> : encode_gpu_layout<4, false>);
        encoded = encode(raw.data(), numel, *table, stream_id,
            bit_reorder, mantissa_bits, exponent_bits);
    }

    nb::list raw_streams;
    for (const auto& stream : encoded.raw_streams) {
        raw_streams.append(bytes_from_vector(stream));
    }
    nb::list positions;
    for (const auto position : encoded.symbols.output_positions) {
        positions.append(position);
    }
    return nb::make_tuple(
        bytes_from_vector(encoded.symbols.bitstream),
        nb::tuple(raw_streams),
        bytes_from_vector(encoded.symbols.packed_gaps),
        nb::tuple(positions),
        encoded.symbols.bit_length,
        encoded.symbols.max_bits);
}
