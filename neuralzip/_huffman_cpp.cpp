#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using CodeLengths = std::array<std::uint8_t, 256>;
using Counts = std::array<std::uint64_t, 256>;
using Bits = std::vector<std::uint8_t>;

struct HeapItem {
    std::uint64_t count;
    int order;
    std::vector<int> symbols;
};

struct HeapCompare {
    bool operator()(const HeapItem& left, const HeapItem& right) const {
        if (left.count != right.count) {
            return left.count > right.count;
        }
        return left.order > right.order;
    }
};

struct DecodeNode {
    int child[2] = {-1, -1};
    int symbol = -1;
};

class BitWriter {
public:
    void write(const Bits& bits) {
        for (std::uint8_t bit : bits) {
            current_ = static_cast<std::uint8_t>((current_ << 1) | bit);
            bit_count_ += 1;
            if (bit_count_ == 8) {
                output_.push_back(current_);
                current_ = 0;
                bit_count_ = 0;
            }
        }
    }

    std::vector<std::uint8_t> finish() {
        if (bit_count_ > 0) {
            output_.push_back(static_cast<std::uint8_t>(current_ << (8 - bit_count_)));
        }
        return std::move(output_);
    }

private:
    std::vector<std::uint8_t> output_;
    std::uint8_t current_ = 0;
    int bit_count_ = 0;
};

void increment_binary(Bits& bits) {
    for (auto index = bits.rbegin(); index != bits.rend(); ++index) {
        if (*index == 0) {
            *index = 1;
            return;
        }
        *index = 0;
    }
    bits.insert(bits.begin(), 1);
}

CodeLengths make_code_lengths(const Counts& counts) {
    CodeLengths lengths{};
    std::priority_queue<HeapItem, std::vector<HeapItem>, HeapCompare> heap;

    for (int symbol = 0; symbol < 256; ++symbol) {
        if (counts[symbol] > 0) {
            heap.push(HeapItem{counts[symbol], symbol, {symbol}});
        }
    }

    if (heap.empty()) {
        return lengths;
    }
    if (heap.size() == 1) {
        lengths[heap.top().symbols.front()] = 1;
        return lengths;
    }

    int next_order = 256;
    while (heap.size() > 1) {
        HeapItem left = heap.top();
        heap.pop();
        HeapItem right = heap.top();
        heap.pop();

        std::vector<int> merged;
        merged.reserve(left.symbols.size() + right.symbols.size());
        for (int symbol : left.symbols) {
            lengths[symbol] += 1;
            merged.push_back(symbol);
        }
        for (int symbol : right.symbols) {
            lengths[symbol] += 1;
            merged.push_back(symbol);
        }

        heap.push(HeapItem{left.count + right.count, next_order++, std::move(merged)});
    }

    return lengths;
}

std::vector<std::pair<int, int>> sorted_symbols(const CodeLengths& lengths) {
    std::vector<std::pair<int, int>> symbols;
    symbols.reserve(256);
    for (int symbol = 0; symbol < 256; ++symbol) {
        if (lengths[symbol] > 0) {
            symbols.push_back({lengths[symbol], symbol});
        }
    }
    std::sort(symbols.begin(), symbols.end());
    return symbols;
}

std::array<Bits, 256> make_canonical_codes(const CodeLengths& lengths) {
    std::array<Bits, 256> codes;
    Bits current_code;

    for (const auto& [length, symbol] : sorted_symbols(lengths)) {
        if (current_code.size() > static_cast<std::size_t>(length)) {
            throw std::runtime_error("Invalid canonical Huffman code lengths.");
        }
        current_code.resize(static_cast<std::size_t>(length), 0);
        codes[symbol] = current_code;
        increment_binary(current_code);
    }

    return codes;
}

std::vector<DecodeNode> make_decode_tree(const CodeLengths& lengths) {
    std::vector<DecodeNode> tree(1);
    auto codes = make_canonical_codes(lengths);

    for (int symbol = 0; symbol < 256; ++symbol) {
        const Bits& bits = codes[symbol];
        if (bits.empty()) {
            continue;
        }

        int node = 0;
        for (std::size_t index = 0; index < bits.size(); ++index) {
            if (tree[node].symbol >= 0) {
                throw std::runtime_error("Invalid prefix in Huffman codebook.");
            }

            int bit = bits[index] ? 1 : 0;
            if (tree[node].child[bit] < 0) {
                tree[node].child[bit] = static_cast<int>(tree.size());
                tree.push_back(DecodeNode{});
            }
            node = tree[node].child[bit];
        }

        if (tree[node].symbol >= 0 || tree[node].child[0] >= 0 || tree[node].child[1] >= 0) {
            throw std::runtime_error("Invalid duplicate or prefix Huffman code.");
        }
        tree[node].symbol = symbol;
    }

    return tree;
}

PyObject* bytes_from_vector(const std::vector<std::uint8_t>& data) {
    const char* pointer = data.empty() ? "" : reinterpret_cast<const char*>(data.data());
    return PyBytes_FromStringAndSize(pointer, static_cast<Py_ssize_t>(data.size()));
}

PyObject* py_compress(PyObject*, PyObject* args, PyObject* kwargs) {
    Py_buffer data_view;
    Py_ssize_t min_saving_bytes = 16;
    static const char* keyword_names[] = {"data", "min_saving_bytes", nullptr};

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "y*|n",
            const_cast<char**>(keyword_names),
            &data_view,
            &min_saving_bytes)) {
        return nullptr;
    }

    if (min_saving_bytes < 0) {
        PyBuffer_Release(&data_view);
        PyErr_SetString(PyExc_ValueError, "min_saving_bytes must be non-negative.");
        return nullptr;
    }

    try {
        if (data_view.len == 0) {
            PyBuffer_Release(&data_view);
            Py_RETURN_NONE;
        }

        Counts counts{};
        const auto* data = static_cast<const std::uint8_t*>(data_view.buf);
        for (Py_ssize_t index = 0; index < data_view.len; ++index) {
            counts[data[index]] += 1;
        }

        CodeLengths lengths = make_code_lengths(counts);
        unsigned __int128 encoded_bits = 0;
        for (int symbol = 0; symbol < 256; ++symbol) {
            encoded_bits += static_cast<unsigned __int128>(counts[symbol]) * lengths[symbol];
        }
        unsigned __int128 encoded_bytes_128 = (encoded_bits + 7) / 8;
        if (encoded_bytes_128 > std::numeric_limits<std::size_t>::max()) {
            PyBuffer_Release(&data_view);
            Py_RETURN_NONE;
        }

        std::size_t encoded_bytes = static_cast<std::size_t>(encoded_bytes_128);
        std::size_t overhead = 256 + static_cast<std::size_t>(min_saving_bytes);
        if (encoded_bytes + overhead >= static_cast<std::size_t>(data_view.len)) {
            PyBuffer_Release(&data_view);
            Py_RETURN_NONE;
        }

        auto codes = make_canonical_codes(lengths);
        BitWriter writer;
        for (Py_ssize_t index = 0; index < data_view.len; ++index) {
            writer.write(codes[data[index]]);
        }
        std::vector<std::uint8_t> encoded = writer.finish();

        if (encoded.size() + overhead >= static_cast<std::size_t>(data_view.len)) {
            PyBuffer_Release(&data_view);
            Py_RETURN_NONE;
        }

        PyObject* payload_object = bytes_from_vector(encoded);
        PyObject* lengths_object = PyBytes_FromStringAndSize(
            reinterpret_cast<const char*>(lengths.data()),
            static_cast<Py_ssize_t>(lengths.size()));
        PyObject* count_object = PyLong_FromSsize_t(data_view.len);
        PyBuffer_Release(&data_view);

        if (!payload_object || !lengths_object || !count_object) {
            Py_XDECREF(payload_object);
            Py_XDECREF(lengths_object);
            Py_XDECREF(count_object);
            return nullptr;
        }

        PyObject* result = PyTuple_Pack(3, payload_object, lengths_object, count_object);
        Py_DECREF(payload_object);
        Py_DECREF(lengths_object);
        Py_DECREF(count_object);
        return result;
    } catch (const std::bad_alloc&) {
        PyBuffer_Release(&data_view);
        PyErr_NoMemory();
        return nullptr;
    } catch (const std::exception& error) {
        PyBuffer_Release(&data_view);
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

PyObject* py_decompress(PyObject*, PyObject* args) {
    Py_buffer payload_view;
    Py_buffer lengths_view;
    Py_ssize_t symbol_count;

    if (!PyArg_ParseTuple(args, "y*y*n", &payload_view, &lengths_view, &symbol_count)) {
        return nullptr;
    }

    if (symbol_count < 0) {
        PyBuffer_Release(&payload_view);
        PyBuffer_Release(&lengths_view);
        PyErr_SetString(PyExc_ValueError, "symbol_count must be non-negative.");
        return nullptr;
    }
    if (lengths_view.len != 256) {
        PyBuffer_Release(&payload_view);
        PyBuffer_Release(&lengths_view);
        PyErr_SetString(PyExc_ValueError, "code_lengths must contain exactly 256 bytes.");
        return nullptr;
    }
    if (symbol_count == 0) {
        PyBuffer_Release(&payload_view);
        PyBuffer_Release(&lengths_view);
        return PyBytes_FromStringAndSize("", 0);
    }

    try {
        CodeLengths lengths{};
        const auto* length_data = static_cast<const std::uint8_t*>(lengths_view.buf);
        for (int index = 0; index < 256; ++index) {
            lengths[index] = length_data[index];
        }

        std::vector<DecodeNode> tree = make_decode_tree(lengths);
        std::vector<std::uint8_t> output;
        output.reserve(static_cast<std::size_t>(symbol_count));

        const auto* payload = static_cast<const std::uint8_t*>(payload_view.buf);
        int node = 0;
        for (Py_ssize_t byte_index = 0; byte_index < payload_view.len; ++byte_index) {
            std::uint8_t byte = payload[byte_index];
            for (int bit_index = 7; bit_index >= 0; --bit_index) {
                int bit = (byte >> bit_index) & 1;
                int next = tree[node].child[bit];
                if (next < 0) {
                    PyBuffer_Release(&payload_view);
                    PyBuffer_Release(&lengths_view);
                    PyErr_SetString(PyExc_ValueError, "Invalid Huffman payload.");
                    return nullptr;
                }

                node = next;
                if (tree[node].symbol >= 0) {
                    output.push_back(static_cast<std::uint8_t>(tree[node].symbol));
                    if (output.size() == static_cast<std::size_t>(symbol_count)) {
                        PyBuffer_Release(&payload_view);
                        PyBuffer_Release(&lengths_view);
                        return bytes_from_vector(output);
                    }
                    node = 0;
                }
            }
        }

        PyBuffer_Release(&payload_view);
        PyBuffer_Release(&lengths_view);
        PyErr_SetString(PyExc_ValueError, "Truncated Huffman payload.");
        return nullptr;
    } catch (const std::bad_alloc&) {
        PyBuffer_Release(&payload_view);
        PyBuffer_Release(&lengths_view);
        PyErr_NoMemory();
        return nullptr;
    } catch (const std::exception& error) {
        PyBuffer_Release(&payload_view);
        PyBuffer_Release(&lengths_view);
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

PyMethodDef methods[] = {
    {"compress", reinterpret_cast<PyCFunction>(py_compress), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"decompress", py_decompress, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_huffman_cpp",
    nullptr,
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__huffman_cpp() {
    return PyModule_Create(&module);
}
