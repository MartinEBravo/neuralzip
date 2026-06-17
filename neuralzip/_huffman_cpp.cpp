#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
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

struct EncoderCodebook {
    CodeLengths lengths{};
    std::array<Bits, 256> codes{};
};

struct DecoderCodebook {
    std::vector<DecodeNode> tree;
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

    bool exceeds(std::size_t max_output_bytes) const {
        std::size_t finished_size = output_.size() + (bit_count_ > 0 ? 1 : 0);
        return finished_size > max_output_bytes;
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

PyObject* code_lengths_to_python(const CodeLengths& lengths) {
    return PyBytes_FromStringAndSize(
        reinterpret_cast<const char*>(lengths.data()),
        static_cast<Py_ssize_t>(lengths.size()));
}

CodeLengths code_lengths_from_view(const Py_buffer& lengths_view) {
    if (lengths_view.len != 256) {
        throw std::runtime_error("code_lengths must contain exactly 256 bytes.");
    }

    CodeLengths lengths{};
    const auto* length_data = static_cast<const std::uint8_t*>(lengths_view.buf);
    for (int index = 0; index < 256; ++index) {
        lengths[index] = length_data[index];
    }
    return lengths;
}

std::vector<std::uint8_t> encode_with_codes(
    const std::uint8_t* data,
    Py_ssize_t data_len,
    const std::array<Bits, 256>& codes,
    std::size_t max_output_bytes = std::numeric_limits<std::size_t>::max()) {
    BitWriter writer;
    for (Py_ssize_t index = 0; index < data_len; ++index) {
        const Bits& code = codes[data[index]];
        if (code.empty()) {
            throw std::runtime_error("Input contains a symbol missing from code_lengths.");
        }
        writer.write(code);
        if (writer.exceeds(max_output_bytes)) {
            return {};
        }
    }
    return writer.finish();
}

std::vector<std::uint8_t> encode_with_lengths(
    const std::uint8_t* data,
    Py_ssize_t data_len,
    const CodeLengths& lengths) {
    auto codes = make_canonical_codes(lengths);
    return encode_with_codes(data, data_len, codes);
}

void destroy_codebook_capsule(PyObject* capsule) {
    void* pointer = PyCapsule_GetPointer(capsule, "neuralzip.EncoderCodebook");
    delete static_cast<EncoderCodebook*>(pointer);
}

EncoderCodebook* codebook_from_capsule(PyObject* capsule) {
    void* pointer = PyCapsule_GetPointer(capsule, "neuralzip.EncoderCodebook");
    if (pointer == nullptr) {
        throw std::runtime_error("Invalid Huffman codebook capsule.");
    }
    return static_cast<EncoderCodebook*>(pointer);
}

void destroy_decodebook_capsule(PyObject* capsule) {
    void* pointer = PyCapsule_GetPointer(capsule, "neuralzip.DecoderCodebook");
    delete static_cast<DecoderCodebook*>(pointer);
}

DecoderCodebook* decodebook_from_capsule(PyObject* capsule) {
    void* pointer = PyCapsule_GetPointer(capsule, "neuralzip.DecoderCodebook");
    if (pointer == nullptr) {
        throw std::runtime_error("Invalid Huffman decodebook capsule.");
    }
    return static_cast<DecoderCodebook*>(pointer);
}

std::vector<std::uint8_t> decode_with_tree(
    const std::uint8_t* payload,
    Py_ssize_t payload_len,
    const std::vector<DecodeNode>& tree,
    Py_ssize_t symbol_count) {
    std::vector<std::uint8_t> output;
    output.reserve(static_cast<std::size_t>(symbol_count));

    int node = 0;
    for (Py_ssize_t byte_index = 0; byte_index < payload_len; ++byte_index) {
        std::uint8_t byte = payload[byte_index];
        for (int bit_index = 7; bit_index >= 0; --bit_index) {
            int bit = (byte >> bit_index) & 1;
            int next = tree[node].child[bit];
            if (next < 0) {
                throw std::runtime_error("Invalid Huffman payload.");
            }

            node = next;
            if (tree[node].symbol >= 0) {
                output.push_back(static_cast<std::uint8_t>(tree[node].symbol));
                if (output.size() == static_cast<std::size_t>(symbol_count)) {
                    return output;
                }
                node = 0;
            }
        }
    }

    throw std::runtime_error("Truncated Huffman payload.");
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

        std::vector<std::uint8_t> encoded = encode_with_lengths(data, data_view.len, lengths);

        if (encoded.size() + overhead >= static_cast<std::size_t>(data_view.len)) {
            PyBuffer_Release(&data_view);
            Py_RETURN_NONE;
        }

        PyObject* payload_object = bytes_from_vector(encoded);
        PyObject* lengths_object = code_lengths_to_python(lengths);
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

PyObject* py_build_code_lengths(PyObject*, PyObject* args, PyObject* kwargs) {
    Py_buffer data_view;
    Py_ssize_t prior_count = 0;
    static const char* keyword_names[] = {"data", "prior_count", nullptr};

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "y*|n",
            const_cast<char**>(keyword_names),
            &data_view,
            &prior_count)) {
        return nullptr;
    }

    if (prior_count < 0) {
        PyBuffer_Release(&data_view);
        PyErr_SetString(PyExc_ValueError, "prior_count must be non-negative.");
        return nullptr;
    }

    try {
        Counts counts{};
        if (prior_count > 0) {
            for (int symbol = 0; symbol < 256; ++symbol) {
                counts[symbol] = static_cast<std::uint64_t>(prior_count);
            }
        }

        const auto* data = static_cast<const std::uint8_t*>(data_view.buf);
        for (Py_ssize_t index = 0; index < data_view.len; ++index) {
            counts[data[index]] += 1;
        }

        CodeLengths lengths = make_code_lengths(counts);
        PyBuffer_Release(&data_view);
        return code_lengths_to_python(lengths);
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

PyObject* py_build_codebook(PyObject*, PyObject* args, PyObject* kwargs) {
    Py_buffer data_view;
    Py_ssize_t prior_count = 0;
    static const char* keyword_names[] = {"data", "prior_count", nullptr};

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "y*|n",
            const_cast<char**>(keyword_names),
            &data_view,
            &prior_count)) {
        return nullptr;
    }

    if (prior_count < 0) {
        PyBuffer_Release(&data_view);
        PyErr_SetString(PyExc_ValueError, "prior_count must be non-negative.");
        return nullptr;
    }

    try {
        Counts counts{};
        if (prior_count > 0) {
            for (int symbol = 0; symbol < 256; ++symbol) {
                counts[symbol] = static_cast<std::uint64_t>(prior_count);
            }
        }

        const auto* data = static_cast<const std::uint8_t*>(data_view.buf);
        for (Py_ssize_t index = 0; index < data_view.len; ++index) {
            counts[data[index]] += 1;
        }

        auto codebook = std::make_unique<EncoderCodebook>();
        codebook->lengths = make_code_lengths(counts);
        codebook->codes = make_canonical_codes(codebook->lengths);

        PyObject* capsule = PyCapsule_New(
            codebook.get(),
            "neuralzip.EncoderCodebook",
            destroy_codebook_capsule);
        PyObject* lengths_object = code_lengths_to_python(codebook->lengths);
        PyBuffer_Release(&data_view);

        if (!capsule || !lengths_object) {
            Py_XDECREF(capsule);
            Py_XDECREF(lengths_object);
            return nullptr;
        }
        codebook.release();

        PyObject* result = PyTuple_Pack(2, capsule, lengths_object);
        Py_DECREF(capsule);
        Py_DECREF(lengths_object);
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

PyObject* py_encode_with_code_lengths(PyObject*, PyObject* args, PyObject* kwargs) {
    Py_buffer data_view;
    Py_buffer lengths_view;
    Py_ssize_t min_saving_bytes = 16;
    Py_ssize_t max_output_bytes = -1;
    static const char* keyword_names[] = {
        "data",
        "code_lengths",
        "min_saving_bytes",
        "max_output_bytes",
        nullptr,
    };

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "y*y*|nn",
            const_cast<char**>(keyword_names),
            &data_view,
            &lengths_view,
            &min_saving_bytes,
            &max_output_bytes)) {
        return nullptr;
    }

    if (min_saving_bytes < 0) {
        PyBuffer_Release(&data_view);
        PyBuffer_Release(&lengths_view);
        PyErr_SetString(PyExc_ValueError, "min_saving_bytes must be non-negative.");
        return nullptr;
    }

    try {
        if (data_view.len == 0) {
            PyBuffer_Release(&data_view);
            PyBuffer_Release(&lengths_view);
            Py_RETURN_NONE;
        }

        CodeLengths lengths = code_lengths_from_view(lengths_view);
        const auto* data = static_cast<const std::uint8_t*>(data_view.buf);
        std::size_t max_bytes =
            max_output_bytes < 0
                ? std::numeric_limits<std::size_t>::max()
                : static_cast<std::size_t>(max_output_bytes);

        std::vector<std::uint8_t> encoded =
            encode_with_codes(data, data_view.len, make_canonical_codes(lengths), max_bytes);
        if (encoded.empty() && data_view.len > 0) {
            PyBuffer_Release(&data_view);
            PyBuffer_Release(&lengths_view);
            Py_RETURN_NONE;
        }
        if (encoded.size() + static_cast<std::size_t>(min_saving_bytes)
            >= static_cast<std::size_t>(data_view.len)) {
            PyBuffer_Release(&data_view);
            PyBuffer_Release(&lengths_view);
            Py_RETURN_NONE;
        }

        PyObject* payload_object = bytes_from_vector(encoded);
        PyObject* count_object = PyLong_FromSsize_t(data_view.len);
        PyBuffer_Release(&data_view);
        PyBuffer_Release(&lengths_view);

        if (!payload_object || !count_object) {
            Py_XDECREF(payload_object);
            Py_XDECREF(count_object);
            return nullptr;
        }

        PyObject* result = PyTuple_Pack(2, payload_object, count_object);
        Py_DECREF(payload_object);
        Py_DECREF(count_object);
        return result;
    } catch (const std::bad_alloc&) {
        PyBuffer_Release(&data_view);
        PyBuffer_Release(&lengths_view);
        PyErr_NoMemory();
        return nullptr;
    } catch (const std::exception& error) {
        PyBuffer_Release(&data_view);
        PyBuffer_Release(&lengths_view);
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

PyObject* py_encode_with_codebook(PyObject*, PyObject* args, PyObject* kwargs) {
    Py_buffer data_view;
    PyObject* capsule = nullptr;
    Py_ssize_t min_saving_bytes = 16;
    Py_ssize_t max_output_bytes = -1;
    static const char* keyword_names[] = {
        "data",
        "codebook",
        "min_saving_bytes",
        "max_output_bytes",
        nullptr,
    };

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "y*O|nn",
            const_cast<char**>(keyword_names),
            &data_view,
            &capsule,
            &min_saving_bytes,
            &max_output_bytes)) {
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

        EncoderCodebook* codebook = codebook_from_capsule(capsule);
        const auto* data = static_cast<const std::uint8_t*>(data_view.buf);
        std::size_t max_bytes =
            max_output_bytes < 0
                ? std::numeric_limits<std::size_t>::max()
                : static_cast<std::size_t>(max_output_bytes);

        std::vector<std::uint8_t> encoded =
            encode_with_codes(data, data_view.len, codebook->codes, max_bytes);
        if (encoded.empty() && data_view.len > 0) {
            PyBuffer_Release(&data_view);
            Py_RETURN_NONE;
        }
        if (encoded.size() + static_cast<std::size_t>(min_saving_bytes)
            >= static_cast<std::size_t>(data_view.len)) {
            PyBuffer_Release(&data_view);
            Py_RETURN_NONE;
        }

        PyObject* payload_object = bytes_from_vector(encoded);
        PyObject* count_object = PyLong_FromSsize_t(data_view.len);
        PyBuffer_Release(&data_view);

        if (!payload_object || !count_object) {
            Py_XDECREF(payload_object);
            Py_XDECREF(count_object);
            return nullptr;
        }

        PyObject* result = PyTuple_Pack(2, payload_object, count_object);
        Py_DECREF(payload_object);
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

PyObject* py_build_decodebook(PyObject*, PyObject* args) {
    Py_buffer lengths_view;

    if (!PyArg_ParseTuple(args, "y*", &lengths_view)) {
        return nullptr;
    }

    try {
        CodeLengths lengths = code_lengths_from_view(lengths_view);
        auto decodebook = std::make_unique<DecoderCodebook>();
        decodebook->tree = make_decode_tree(lengths);
        PyBuffer_Release(&lengths_view);

        PyObject* capsule = PyCapsule_New(
            decodebook.get(),
            "neuralzip.DecoderCodebook",
            destroy_decodebook_capsule);
        if (!capsule) {
            return nullptr;
        }
        decodebook.release();
        return capsule;
    } catch (const std::bad_alloc&) {
        PyBuffer_Release(&lengths_view);
        PyErr_NoMemory();
        return nullptr;
    } catch (const std::exception& error) {
        PyBuffer_Release(&lengths_view);
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
        CodeLengths lengths = code_lengths_from_view(lengths_view);
        std::vector<DecodeNode> tree = make_decode_tree(lengths);
        const auto* payload = static_cast<const std::uint8_t*>(payload_view.buf);
        std::vector<std::uint8_t> output =
            decode_with_tree(payload, payload_view.len, tree, symbol_count);

        PyBuffer_Release(&payload_view);
        PyBuffer_Release(&lengths_view);
        return bytes_from_vector(output);
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

PyObject* py_decompress_with_decodebook(PyObject*, PyObject* args) {
    Py_buffer payload_view;
    PyObject* capsule = nullptr;
    Py_ssize_t symbol_count;

    if (!PyArg_ParseTuple(args, "y*On", &payload_view, &capsule, &symbol_count)) {
        return nullptr;
    }

    if (symbol_count < 0) {
        PyBuffer_Release(&payload_view);
        PyErr_SetString(PyExc_ValueError, "symbol_count must be non-negative.");
        return nullptr;
    }
    if (symbol_count == 0) {
        PyBuffer_Release(&payload_view);
        return PyBytes_FromStringAndSize("", 0);
    }

    try {
        DecoderCodebook* decodebook = decodebook_from_capsule(capsule);
        const auto* payload = static_cast<const std::uint8_t*>(payload_view.buf);
        std::vector<std::uint8_t> output =
            decode_with_tree(payload, payload_view.len, decodebook->tree, symbol_count);
        PyBuffer_Release(&payload_view);
        return bytes_from_vector(output);
    } catch (const std::bad_alloc&) {
        PyBuffer_Release(&payload_view);
        PyErr_NoMemory();
        return nullptr;
    } catch (const std::exception& error) {
        PyBuffer_Release(&payload_view);
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

PyMethodDef methods[] = {
    {"compress", reinterpret_cast<PyCFunction>(py_compress), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"build_code_lengths", reinterpret_cast<PyCFunction>(py_build_code_lengths), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"build_codebook", reinterpret_cast<PyCFunction>(py_build_codebook), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"build_decodebook", py_build_decodebook, METH_VARARGS, nullptr},
    {"encode_with_code_lengths", reinterpret_cast<PyCFunction>(py_encode_with_code_lengths), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"encode_with_codebook", reinterpret_cast<PyCFunction>(py_encode_with_codebook), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"decompress", py_decompress, METH_VARARGS, nullptr},
    {"decompress_with_decodebook", py_decompress_with_decodebook, METH_VARARGS, nullptr},
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
