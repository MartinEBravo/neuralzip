std::unique_ptr<Huff0Codebook> make_codebook_from_counts(
    const std::array<unsigned, 256>& counts,
    unsigned max_bits) {
    unsigned symbol_count = 0;
    unsigned max_symbol_value = 0;
    for (unsigned symbol = 0; symbol < counts.size(); ++symbol) {
        if (counts[symbol] > 0) {
            ++symbol_count;
            max_symbol_value = symbol;
        }
    }
    require(symbol_count >= 2, "Huff0 requires at least two symbols.");
    require(
        symbol_count <= (1U << max_bits),
        "Huff0 max_bits cannot represent every observed symbol.");
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

std::unique_ptr<Huff0Codebook> make_codebook_from_header(
    const std::uint8_t* header,
    Py_ssize_t header_len,
    const std::uint8_t* vocab = nullptr,
    std::size_t vocab_size = 0) {
    require(header_len > 0, "Huff0 codebook header is empty.");

    auto codebook = std::make_unique<Huff0Codebook>();
    if (is_mixed_header(header, header_len)) {
        codebook->mixed = make_mixed_table(header, header_len, vocab, vocab_size);
        codebook->serialized.assign(header, header + header_len);
        return codebook;
    }
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
    require(!codebook->mixed, "GPU export requires a singleton Huff0 table.");
    require(
        max_bits > 0 && max_bits <= HUF_TABLELOG_ABSOLUTEMAX,
        "max_bits must be in [1, HUF_TABLELOG_ABSOLUTEMAX].");

    const HUF_CTableHeader table_header = HUF_readCTableHeader(
        codebook->ctable.data());
    require(
        table_header.tableLog <= max_bits,
        "Huff0 codebook exceeds the requested max_bits limit.");

    constexpr unsigned ctable_word_bits = sizeof(HUF_CElt) * 8U;
    ExportedCodeTable output;
    output.max_bits = table_header.tableLog;
    for (unsigned symbol = 0; symbol <= kMaxSymbolValue; ++symbol) {
        const unsigned length = HUF_getNbBitsFromCTable(
            codebook->ctable.data(), symbol);
        output.code_lengths[symbol] = static_cast<std::uint8_t>(length);
        if (length == 0) {
            continue;
        }
        const HUF_CElt entry = codebook->ctable[
            static_cast<std::size_t>(symbol) + 1U];
        output.code_values[symbol] = static_cast<std::uint16_t>(
            entry >> (ctable_word_bits - length));
    }
    return output;
}


std::size_t encode_huff0_block_into(
    const std::uint8_t* data,
    std::size_t data_size,
    const Huff0Codebook& codebook,
    std::uint8_t* output,
    std::size_t output_capacity,
    std::size_t max_output_bytes) {
    if (data_size == 0 || data_size > kMaxBlockSize || output_capacity == 0) {
        return 0;
    }
    if (!codebook.covers_all_symbols) {
        for (std::size_t index = 0; index < data_size; ++index) {
            if (HUF_getNbBitsFromCTable(codebook.ctable.data(), data[index]) == 0) {
                return 0;
            }
        }
    }
    const std::size_t size = HUF_compress4X_usingCTable(
        output,
        output_capacity,
        data,
        data_size,
        codebook.ctable.data(),
        kFlags);
    if (HUF_isError(size) || size == 0 || size > max_output_bytes) {
        return 0;
    }
    return size;
}

std::unique_ptr<Huff0Decodebook> make_decodebook_from_header(
    const std::uint8_t* header,
    Py_ssize_t header_len,
    const std::uint8_t* vocab = nullptr,
    std::size_t vocab_size = 0) {
    require(header_len > 0, "Huff0 codebook header is empty.");

    auto decodebook = std::make_unique<Huff0Decodebook>();
    if (is_mixed_header(header, header_len)) {
        decodebook->mixed = make_mixed_table(header, header_len, vocab, vocab_size);
        return decodebook;
    }
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
    const size_t read_size_x2 = HUF_readDTableX2_wksp(
        decodebook->dtable_x2.data(),
        header,
        static_cast<std::size_t>(header_len),
        workspace.data(),
        workspace.size() * sizeof(unsigned),
        kFlags);
    check_huf(read_size_x2);
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
    if (symbol_count == 0) {
        return;
    }
    const HUF_DTable* dtable = select_decode_dtable(
        decodebook, symbol_count, static_cast<std::size_t>(payload_len));
    const size_t decoded = HUF_decompress4X_usingDTable(
        output, symbol_count, payload,
        static_cast<std::size_t>(payload_len), dtable, kFlags);
    check_huf(decoded);
    require(decoded == symbol_count, "Huff0 decoded size mismatch.");
}
