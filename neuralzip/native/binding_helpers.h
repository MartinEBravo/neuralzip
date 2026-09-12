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

nb::tuple build_codebook_from_header(nb::handle header_source) {
    ReadOnlyBuffer header(header_source);
    return codebook_to_python(make_codebook_from_header(
        header.data(), header.size()));
}

nb::capsule build_decodebook(nb::handle header_source) {
    ReadOnlyBuffer header(header_source);
    auto decodebook = make_decodebook_from_header(
        header.data(), header.size());
    Huff0Decodebook* native = decodebook.get();
    nb::capsule handle(native, "neuralzip.Huff0Decodebook", delete_capsule<Huff0Decodebook>);
    decodebook.release();
    return handle;
}

nb::tuple export_code_table(
    nb::handle header_source,
    Py_ssize_t max_bits) {
    require<nb::value_error>(
        max_bits > 0 && max_bits <= 12,
        "max_bits must be in [1, 12].");
    ReadOnlyBuffer header(header_source);
    const ExportedCodeTable table = export_code_table_from_header(
        header.data(), header.size(), static_cast<unsigned>(max_bits));
    nb::list values;
    for (const auto value : table.code_values) {
        values.append(value);
    }
    return nb::make_tuple(
        nb::bytes(table.code_lengths.data(), table.code_lengths.size()),
        nb::tuple(values),
        table.max_bits);
}

nb::tuple encode_gpu_bfloat16(
    nb::handle raw_source,
    nb::handle header_source,
    Py_ssize_t bytes_per_thread,
    Py_ssize_t threads_per_block) {
    require<nb::value_error>(
        bytes_per_thread == 8 && threads_per_block == 512,
        "The GPU ABI requires 8 bytes per thread and 512 threads per block.");
    ReadOnlyBuffer raw(raw_source);
    ReadOnlyBuffer header(header_source);

    GPUBFloat16Encoding encoded;
    {
        nb::gil_scoped_release release;
        encoded = encode_gpu_bfloat16_layout(
            raw.data(),
            static_cast<std::size_t>(raw.size()),
            header.data(),
            header.size());
    }

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
