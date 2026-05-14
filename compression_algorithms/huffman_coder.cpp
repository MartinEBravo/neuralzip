#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct Code {
    uint64_t bits = 0;
    uint8_t length = 0;
};

struct TreeNode {
    uint64_t frequency = 0;
    uint32_t min_symbol = 0;
    int left = -1;
    int right = -1;
};

using HeapItem = std::tuple<uint64_t, uint32_t, int>;

void assign_lengths(
    const std::vector<TreeNode>& nodes,
    int node_index,
    uint8_t depth,
    std::unordered_map<uint32_t, uint8_t>* lengths
) {
    const TreeNode& node = nodes[node_index];
    if (node.left < 0 && node.right < 0) {
        if (depth == 0) {
            depth = 1;
        }
        if (depth > 64) {
            throw std::runtime_error("huffman code length exceeds native width");
        }
        (*lengths)[node.min_symbol] = depth;
        return;
    }
    assign_lengths(nodes, node.left, depth + 1, lengths);
    assign_lengths(nodes, node.right, depth + 1, lengths);
}

std::unordered_map<uint32_t, uint8_t> build_code_lengths(
    const uint32_t* values,
    const uint32_t* counts,
    size_t value_count
) {
    std::unordered_map<uint32_t, uint8_t> lengths;
    lengths.reserve(value_count * 2 + 1);
    if (value_count == 0) {
        return lengths;
    }
    if (value_count == 1) {
        lengths[values[0]] = 1;
        return lengths;
    }

    std::vector<TreeNode> nodes;
    nodes.reserve(value_count * 2);
    std::priority_queue<HeapItem, std::vector<HeapItem>, std::greater<HeapItem>> heap;
    for (size_t i = 0; i < value_count; ++i) {
        nodes.push_back(TreeNode{counts[i], values[i], -1, -1});
        heap.emplace(counts[i], values[i], static_cast<int>(nodes.size() - 1));
    }

    while (heap.size() > 1) {
        auto [left_freq, left_min, left_idx] = heap.top();
        heap.pop();
        auto [right_freq, right_min, right_idx] = heap.top();
        heap.pop();
        uint32_t min_symbol = std::min(left_min, right_min);
        uint64_t frequency = left_freq + right_freq;
        nodes.push_back(TreeNode{frequency, min_symbol, left_idx, right_idx});
        heap.emplace(frequency, min_symbol, static_cast<int>(nodes.size() - 1));
    }

    auto [root_freq, root_min, root_idx] = heap.top();
    (void)root_freq;
    (void)root_min;
    assign_lengths(nodes, root_idx, 0, &lengths);
    return lengths;
}

struct BitWriter {
    std::vector<uint8_t> bytes;
    uint8_t current = 0;
    int bits_filled = 0;

    void write_bits(uint64_t bits, uint8_t length) {
        for (int shift = int(length) - 1; shift >= 0; --shift) {
            current = static_cast<uint8_t>((current << 1) | ((bits >> shift) & 1ULL));
            bits_filled += 1;
            if (bits_filled == 8) {
                bytes.push_back(current);
                current = 0;
                bits_filled = 0;
            }
        }
    }

    std::vector<uint8_t> finish() {
        if (bits_filled != 0) {
            bytes.push_back(static_cast<uint8_t>(current << (8 - bits_filled)));
            current = 0;
            bits_filled = 0;
        }
        return bytes;
    }
};

struct BitReader {
    const uint8_t* data;
    size_t size;
    size_t byte_index = 0;
    int bit_index = 0;

    uint32_t read_bit() {
        if (byte_index >= size) {
            return 0;
        }
        uint8_t byte = data[byte_index];
        uint32_t bit = (byte >> (7 - bit_index)) & 1U;
        bit_index += 1;
        if (bit_index == 8) {
            bit_index = 0;
            byte_index += 1;
        }
        return bit;
    }
};

std::unordered_map<uint32_t, Code> build_encode_table(
    const uint32_t* values,
    const uint8_t* lengths,
    size_t value_count
) {
    std::vector<std::pair<uint8_t, uint32_t>> ordered;
    ordered.reserve(value_count);
    for (size_t i = 0; i < value_count; ++i) {
        if (lengths[i] == 0 || lengths[i] > 64) {
            throw std::runtime_error("invalid code length");
        }
        ordered.emplace_back(lengths[i], values[i]);
    }
    std::sort(ordered.begin(), ordered.end());

    std::unordered_map<uint32_t, Code> table;
    table.reserve(value_count * 2 + 1);
    uint64_t code = 0;
    uint8_t previous_length = 0;
    for (const auto& item : ordered) {
        uint8_t length = item.first;
        uint32_t symbol = item.second;
        if (length > previous_length) {
            code <<= (length - previous_length);
        }
        table[symbol] = Code{code, length};
        code += 1;
        previous_length = length;
    }
    return table;
}

std::vector<std::unordered_map<uint64_t, uint32_t>> build_decode_table(
    const uint32_t* values,
    const uint8_t* lengths,
    size_t value_count,
    uint8_t* max_length
) {
    auto encode_table = build_encode_table(values, lengths, value_count);
    *max_length = 0;
    std::vector<std::unordered_map<uint64_t, uint32_t>> table(65);
    for (const auto& item : encode_table) {
        uint32_t symbol = item.first;
        Code code = item.second;
        table[code.length][code.bits] = symbol;
        *max_length = std::max(*max_length, code.length);
    }
    return table;
}

}  // namespace

extern "C" int huffman_code_lengths(
    const uint32_t* values,
    const uint32_t* counts,
    size_t value_count,
    uint8_t* out_lengths
) {
    if ((!values && value_count) || (!counts && value_count) ||
        (!out_lengths && value_count)) {
        return 1;
    }

    try {
        auto lengths = build_code_lengths(values, counts, value_count);
        for (size_t i = 0; i < value_count; ++i) {
            auto it = lengths.find(values[i]);
            if (it == lengths.end()) {
                return 2;
            }
            out_lengths[i] = it->second;
        }
        return 0;
    } catch (...) {
        return 3;
    }
}

extern "C" int huffman_encode(
    const uint32_t* symbols,
    size_t symbol_count,
    const uint32_t* values,
    const uint8_t* lengths,
    size_t value_count,
    uint8_t** out_bytes,
    size_t* out_size
) {
    if (!out_bytes || !out_size || (!symbols && symbol_count) ||
        (!values && value_count) || (!lengths && value_count)) {
        return 1;
    }

    try {
        auto table = build_encode_table(values, lengths, value_count);
        BitWriter writer;
        for (size_t i = 0; i < symbol_count; ++i) {
            auto it = table.find(symbols[i]);
            if (it == table.end()) {
                return 2;
            }
            writer.write_bits(it->second.bits, it->second.length);
        }

        std::vector<uint8_t> encoded = writer.finish();
        *out_size = encoded.size();
        *out_bytes = static_cast<uint8_t*>(std::malloc(*out_size));
        if (!*out_bytes && *out_size != 0) {
            return 3;
        }
        if (*out_size != 0) {
            std::memcpy(*out_bytes, encoded.data(), *out_size);
        }
        return 0;
    } catch (...) {
        return 4;
    }
}

extern "C" int huffman_decode(
    const uint8_t* data,
    size_t data_size,
    const uint32_t* values,
    const uint8_t* lengths,
    size_t value_count,
    size_t symbol_count,
    uint32_t** out_symbols
) {
    if (!out_symbols || (!data && data_size) || (!values && value_count) ||
        (!lengths && value_count)) {
        return 1;
    }

    try {
        uint8_t max_length = 0;
        auto table = build_decode_table(values, lengths, value_count, &max_length);
        *out_symbols = static_cast<uint32_t*>(std::malloc(symbol_count * sizeof(uint32_t)));
        if (!*out_symbols && symbol_count != 0) {
            return 2;
        }

        BitReader reader{data, data_size};
        for (size_t idx = 0; idx < symbol_count; ++idx) {
            uint64_t code = 0;
            bool found = false;
            for (uint8_t length = 1; length <= max_length; ++length) {
                code = (code << 1) | reader.read_bit();
                auto it = table[length].find(code);
                if (it != table[length].end()) {
                    (*out_symbols)[idx] = it->second;
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::free(*out_symbols);
                *out_symbols = nullptr;
                return 3;
            }
        }
        return 0;
    } catch (...) {
        return 4;
    }
}

extern "C" void huffman_free(void* ptr) {
    std::free(ptr);
}
