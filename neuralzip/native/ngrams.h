void restore_ngram_tokens_into(
    const std::vector<std::uint8_t>& tokens,
    int n,
    const std::vector<std::uint64_t>& vocab,
    const std::uint8_t* escapes,
    Py_ssize_t escape_len,
    const std::uint8_t* tail,
    Py_ssize_t tail_len,
    std::uint8_t* output,
    std::size_t output_size) {
    const std::size_t expected_size =
        tokens.size() * static_cast<std::size_t>(n)
        + static_cast<std::size_t>(tail_len);
    require(
        expected_size == output_size,
        "Packed n-gram decoded size mismatch.");

    std::size_t escape_offset = 0;
    std::size_t output_offset = 0;
    for (std::uint8_t token : tokens) {
        if (token == 255) {
            std::size_t end = escape_offset + static_cast<std::size_t>(n);
            require(
                end <= static_cast<std::size_t>(escape_len),
                "Packed n-gram escape stream is truncated.");
            std::memcpy(output + output_offset, escapes + escape_offset, n);
            escape_offset = end;
        } else {
            require(
                static_cast<std::size_t>(token) < vocab.size(),
                "Packed n-gram token is outside the vocabulary.");
            std::uint64_t symbol = vocab[token];
            for (int offset = 0; offset < n; ++offset) {
                output[output_offset + static_cast<std::size_t>(offset)] =
                    static_cast<std::uint8_t>(
                        (symbol >> (8 * offset)) & 0xFFU);
            }
        }
        output_offset += static_cast<std::size_t>(n);
    }
    if (tail_len > 0) {
        std::memcpy(
            output + output_offset, tail, static_cast<std::size_t>(tail_len));
    }
}

std::vector<std::uint64_t> vocab_from_bytes(
    const std::uint8_t* data,
    std::size_t size,
    int n) {
    const Py_ssize_t symbol_bytes = n <= 4 ? 4 : 8;
    std::vector<std::uint64_t> vocab(
        size / static_cast<std::size_t>(symbol_bytes));
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
    return vocab;
}

void decode_mixed_stream_chunk_into(
    const MixedStreamDecodeChunk& chunk,
    std::uint8_t* destination) {
    if (chunk.mode == MixedMode::Raw) {
        if (chunk.symbol_count > 0) {
            std::memcpy(
                destination, chunk.payload.data(), chunk.symbol_count);
        }
        return;
    }
    if (chunk.mode == MixedMode::SharedHuff0) {
        decode_huff0_block_into(
            chunk.payload.data(),
            chunk.payload.size(),
            destination,
            chunk.symbol_count,
            *chunk.decodebook);
        return;
    }

    const std::size_t token_count = chunk.symbol_count
        / static_cast<std::size_t>(chunk.ngram_n);
    std::vector<std::uint8_t> tokens(token_count);
    decode_huff0_block_into(
        chunk.payload.data(),
        chunk.payload.size(),
        tokens.data(),
        token_count,
        *chunk.decodebook);
    restore_ngram_tokens_into(
        tokens,
        chunk.ngram_n,
        chunk.vocab,
        chunk.escapes.data(),
        chunk.escapes.size(),
        chunk.tail.data(),
        chunk.tail.size(),
        destination,
        chunk.symbol_count);
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
    const std::vector<std::uint64_t>& vocab,
    int n) {
    NGramVocabLookup lookup;
    lookup.ngram_n = n;
    if (n == 2) {
        lookup.tokens.assign(1U << 16U, 255);
        for (std::size_t token = 0; token < vocab.size(); ++token) {
            const std::uint64_t symbol = vocab[token];
            if (symbol < lookup.tokens.size() && lookup.tokens[symbol] == 255) {
                lookup.tokens[symbol] = static_cast<std::uint8_t>(token);
            }
        }
        return lookup;
    }
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

std::shared_ptr<const NGramVocabLookup> prepare_ngram_vocab_lookup(
    Huff0Codebook& codebook,
    nb::handle vocab_source,
    int n) {
    require<nb::value_error>(
        n >= 2 && n <= 8, "packed_huff0 requires ngram_n in [2, 8].");
    ReadOnlyBuffer vocab(vocab_source);
    const auto size = static_cast<std::size_t>(vocab.size());
    // Called with the GIL held. Workers only read immutable shared owners,
    // including when another call replaces this codebook's cached vocabulary.
    const auto& cached = codebook.ngram_lookup;
    if (cached && cached->ngram_n == n && cached->vocabulary.size() == size
        && (size == 0
            || std::memcmp(cached->vocabulary.data(), vocab.data(), size) == 0)) {
        return cached;
    }
    auto lookup = std::make_shared<NGramVocabLookup>(build_ngram_vocab_lookup(
        vocab_from_bytes(vocab.data(), size, n), n));
    if (size != 0) {
        lookup->vocabulary.assign(vocab.data(), vocab.data() + size);
    }
    codebook.ngram_lookup = lookup;
    return lookup;
}

std::uint8_t find_ngram_token(
    const NGramVocabLookup& lookup,
    std::uint64_t symbol) {
    if (lookup.tokens.empty()) {
        return 255;
    }
    if (lookup.ngram_n == 2) {
        return lookup.tokens[static_cast<std::uint16_t>(symbol)];
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

NGramTokenization tokenize_byte_group_stream_ngrams(
    const std::uint8_t* raw,
    int element_size,
    bool bit_reorder,
    int stream,
    std::size_t start,
    Py_ssize_t data_len,
    int n,
    const NGramVocabLookup& lookup) {
    require(
        n >= 1 && n <= 8,
        "Huff0 n-gram tokenization supports n in [1, 8].");
    require(!lookup.tokens.empty(), "ngram vocabulary lookup is empty.");
    require(data_len >= 0, "data length must be non-negative.");

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
            continue;
        }

        result.tokens.push_back(255);
        result.escapes.insert(
            result.escapes.end(), values.begin(), values.begin() + n);
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
    std::size_t max_output_bytes) {
    NGramBlockResult result;
    if (tokens.tokens.empty()) {
        return result;
    }

    result.payload = encode_huff0_block(
        tokens.tokens.data(),
        static_cast<Py_ssize_t>(tokens.tokens.size()),
        codebook,
        max_output_bytes);
    if (result.payload.empty()
        || result.payload.size() + static_cast<std::size_t>(min_saving_bytes)
            >= tokens.tokens.size()) {
        result.payload.clear();
        return result;
    }

    result.escapes = std::move(tokens.escapes);
    result.tail = std::move(tokens.tail);
    return result;
}

const char* mixed_mode_name(MixedMode mode) {
    static constexpr const char* names[] = {"raw", "shared_huff0", "packed_huff0"};
    return names[static_cast<unsigned>(mode)];
}
