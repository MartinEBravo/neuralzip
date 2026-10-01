bool is_mixed_header(const std::uint8_t* header, std::size_t size) {
    return size >= 4 && std::memcmp(header, "NZM1", 4) == 0;
}

std::shared_ptr<const MixedHuffmanTable> make_mixed_table(
    const std::uint8_t* header, std::size_t size,
    const std::uint8_t* vocab, std::size_t vocab_size, bool encoding) {
    require(size >= 260 && size <= 514 && vocab_size == 2 * (size - 260),
        "Mixed table must contain 256 singletons and at most 254 pairs.");
    auto table = std::make_shared<MixedHuffmanTable>();
    table->lengths.assign(header + 4, header + size);
    if (encoding) {
        table->codes.resize(table->lengths.size());
        table->pair_tokens.assign(65536, 0);
    }
    for (std::size_t i = 0; i < vocab_size; i += 2) {
        const auto pair = static_cast<std::uint16_t>(vocab[i] | (vocab[i + 1] << 8));
        if (encoding) table->pair_tokens[pair] = static_cast<std::uint16_t>(256 + i / 2);
        table->vocabulary.push_back(pair);
    }
    auto sorted_pairs = table->vocabulary;
    std::sort(sorted_pairs.begin(), sorted_pairs.end());
    require(std::adjacent_find(sorted_pairs.begin(), sorted_pairs.end()) == sorted_pairs.end(),
        "Duplicate mixed pair definition.");
    std::vector<std::size_t> order(table->lengths.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        require(table->lengths[i] > 0 && table->lengths[i] <= 56,
            "Mixed code lengths must be in [1, 56].");
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](auto a, auto b) {
        return table->lengths[a] < table->lengths[b]
            || (table->lengths[a] == table->lengths[b] && a < b);
    });
    std::uint64_t code = 0;
    unsigned previous = 0;
    table->nodes.emplace_back();
    for (auto symbol : order) {
        const unsigned length = table->lengths[symbol];
        code <<= length - previous;
        require(code < (std::uint64_t{1} << length), "Oversubscribed mixed Huffman table.");
        if (encoding) table->codes[symbol] = code;
        int node = 0;
        for (int bit = static_cast<int>(length) - 1; bit >= 0; --bit) {
            require(table->nodes[node].symbol < 0, "Mixed Huffman prefix collision.");
            const unsigned branch = (code >> bit) & 1U;
            if (table->nodes[node].children[branch] < 0) {
                table->nodes[node].children[branch] = static_cast<int>(table->nodes.size());
                table->nodes.emplace_back();
            }
            node = table->nodes[node].children[branch];
        }
        table->nodes[node].symbol = static_cast<int>(symbol);
        ++code;
        previous = length;
    }
    require(code == (std::uint64_t{1} << previous), "Incomplete mixed Huffman table.");
    if (encoding) return table;
    table->fast.resize(4096);
    for (unsigned prefix = 0; prefix < table->fast.size(); ++prefix) {
        int node = 0;
        unsigned depth = 0, bytes = 0;
        std::uint32_t output = 0;
        for (unsigned tokens = 1; tokens <= 2;) {
            const auto symbol = table->nodes[node].symbol;
            if (symbol >= 0) {
                const unsigned length = symbol < 256 ? 1 : 2;
                const unsigned value = symbol < 256 ? symbol : table->vocabulary[symbol - 256];
                output |= value << (8U * bytes);
                bytes += length;
                table->fast[prefix] = output | (static_cast<std::uint64_t>(depth) << 32U)
                    | (static_cast<std::uint64_t>(bytes) << 40U)
                    | (static_cast<std::uint64_t>(tokens++) << 48U);
                if (tokens > 2) break;
                node = 0;
            }
            if (depth == 12) break;
            node = table->nodes[node].children[(prefix >> (11U - depth++)) & 1U];
            require(node >= 0, "Incomplete mixed Huffman prefix.");
        }
    }
    return table;
}

template <typename Emit>
void parse_mixed_tokens(const std::uint8_t* data, std::size_t size,
    const std::vector<std::uint16_t>& lookup, Emit&& emit) {
    for (std::size_t i = 0; i < size;) {
        const auto pair = i + 1 < size
            ? lookup[data[i] | (static_cast<unsigned>(data[i + 1]) << 8U)] : 0;
        emit(pair ? pair : data[i]);
        i += pair ? 2 : 1;
    }
}

nb::tuple count_mixed_tokens(nb::handle chunks_source, nb::handle vocab_source) {
    ReadOnlyBuffer vocab(vocab_source);
    require(vocab.size() % 2 == 0 && vocab.size() <= 508, "Invalid mixed pair vocabulary.");
    std::vector<std::uint16_t> lookup(65536, 0);
    for (Py_ssize_t i = 0; i < vocab.size(); i += 2) {
        const unsigned pair = vocab.data()[i] | (vocab.data()[i + 1] << 8U);
        require(lookup[pair] == 0, "Duplicate mixed pair definition.");
        lookup[pair] = static_cast<std::uint16_t>(256 + i / 2);
    }
    std::vector<std::uint64_t> counts(256 + vocab.size() / 2, 0);
    for (auto source : nb::tuple(chunks_source)) {
        ReadOnlyBuffer data(source);
        nb::gil_scoped_release release;
        parse_mixed_tokens(data.data(), data.size(), lookup,
            [&](auto token) { ++counts[token]; });
    }
    nb::list output;
    for (auto count : counts) output.append(count);
    return nb::tuple(output);
}

void write_u32(std::uint8_t* out, std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) out[i] = static_cast<std::uint8_t>(value >> (8U * i));
}

std::uint32_t read_u32(const std::uint8_t* in) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i < 4; ++i) value |= static_cast<std::uint32_t>(in[i]) << (8U * i);
    return value;
}

std::size_t encode_mixed_block_into(const std::uint8_t* data, std::size_t size,
    const MixedHuffmanTable& table, std::uint8_t* out, std::size_t capacity) {
    if (capacity <= 32) return 0;
    std::size_t cursor = 32, source = 0;
    // Split by source positions without resetting the greedy parse at lane boundaries.
    for (unsigned lane = 0; lane < 4; ++lane) {
        const std::size_t end = size * (lane + 1) / 4;
        std::size_t tokens = 0;
        std::uint64_t buffer = 0;
        unsigned pending = 0;
        std::uint64_t bits = 0;
        auto flush = [&]() {
            const unsigned bytes = pending / 8;
            if (bytes > capacity - cursor) return false;
            if (capacity - cursor >= 8) {
                const auto word = buffer << (64 - pending);
                for (unsigned byte = 0; byte < 8; ++byte)
                    out[cursor + byte] = static_cast<std::uint8_t>(word >> (56 - 8 * byte));
                cursor += bytes;
                pending %= 8;
            } else {
                while (pending >= 8) {
                    pending -= 8;
                    out[cursor++] = static_cast<std::uint8_t>(buffer >> pending);
                }
            }
            return true;
        };
        while (source < end) {
            const auto pair = source + 1 < size
                ? table.pair_tokens[data[source] | (static_cast<unsigned>(data[source + 1]) << 8U)] : 0;
            const auto token = pair ? pair : data[source];
            source += pair ? 2 : 1;
            const unsigned length = table.lengths[token];
            if (pending + length > 64 && !flush()) return 0;
            buffer = (buffer << length) | table.codes[token];
            pending += length;
            bits += length;
            ++tokens;
            if (pending >= 32 && !flush()) return 0;
        }
        if (pending >= 8 && !flush()) return 0;
        if (pending) {
            if (cursor == capacity) return 0;
            out[cursor++] = static_cast<std::uint8_t>(buffer << (8 - pending));
        }
        require(bits <= UINT32_MAX && tokens <= UINT32_MAX, "Mixed frame exceeds uint32 limits.");
        write_u32(out + 8 * lane, static_cast<std::uint32_t>(tokens));
        write_u32(out + 8 * lane + 4, static_cast<std::uint32_t>(bits));
    }
    return cursor;
}

void decode_mixed_block_into(const std::uint8_t* data, std::size_t size,
    const MixedHuffmanTable& table, std::uint8_t* out, std::size_t expected) {
    require(size >= 32, "Truncated mixed Huffman frame.");
    std::size_t cursor = 32, written = 0;
    for (unsigned lane = 0; lane < 4; ++lane) {
        const auto count = read_u32(data + 8 * lane);
        const std::uint64_t bit_size = read_u32(data + 8 * lane + 4);
        const std::size_t bytes = (bit_size + 7) / 8;
        require(bytes <= size - cursor && count <= expected - written,
            "Mixed frame lengths exceed its payload.");
        const std::uint8_t* stream = data + cursor;
        std::size_t loaded = 0;
        std::uint64_t buffer = 0, consumed = 0;
        unsigned pending = 0;
        for (std::uint32_t i = 0; i < count;) {
            while (pending < 12 && loaded < bytes) {
                buffer = (buffer << 8) | stream[loaded++];
                pending += 8;
            }
            const unsigned prefix = pending >= 12 ? (buffer >> (pending - 12)) & 4095U
                : (buffer << (12 - pending)) & 4095U;
            const auto entry = table.fast[prefix];
            const unsigned length = (entry >> 32U) & 255U;
            const unsigned token_count = (entry >> 48U) & 255U;
            if (entry && token_count <= count - i && length <= pending && consumed + length <= bit_size) {
                const unsigned byte_count = (entry >> 40U) & 255U;
                require(byte_count <= expected - written, "Mixed expansion exceeds exponent count.");
                const auto value = static_cast<std::uint32_t>(entry);
                for (unsigned byte = 0; byte < byte_count; ++byte)
                    out[written++] = static_cast<std::uint8_t>(value >> (8U * byte));
                pending -= length;
                consumed += length;
                i += token_count;
            } else {
                int node = 0;
                while (table.nodes[node].symbol < 0) {
                    if (!pending && loaded < bytes) { buffer = stream[loaded++]; pending = 8; }
                    require(pending && consumed < bit_size, "Truncated mixed Huffman code.");
                    node = table.nodes[node].children[(buffer >> --pending) & 1U];
                    ++consumed;
                    require(node >= 0, "Invalid mixed Huffman prefix.");
                }
                const unsigned symbol = table.nodes[node].symbol;
                const std::size_t expanded = symbol < 256 ? 1 : 2;
                require(expanded <= expected - written, "Mixed expansion exceeds exponent count.");
                if (expanded == 1) out[written++] = static_cast<std::uint8_t>(symbol);
                else {
                    const auto pair = table.vocabulary[symbol - 256];
                    out[written++] = static_cast<std::uint8_t>(pair);
                    out[written++] = static_cast<std::uint8_t>(pair >> 8U);
                }
                ++i;
            }
        }
        require(consumed == bit_size, "Mixed frame has unconsumed code bits.");
        if (bit_size % 8) require((stream[bytes - 1] & ((1U << (8 - bit_size % 8)) - 1)) == 0,
            "Mixed frame has nonzero padding.");
        cursor += bytes;
    }
    require(cursor == size && written == expected, "Mixed decoded size mismatch.");
}
