#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

extern "C" {
#include "huf.h"
}

namespace {

constexpr std::size_t kMaxBlockSize = HUF_BLOCKSIZE_MAX;
constexpr unsigned kMaxSymbolValue = HUF_SYMBOLVALUE_MAX;
constexpr unsigned kTableLog = HUF_TABLELOG_DEFAULT;
constexpr int kFlags = HUF_flags_disableAsm;

struct Huff0Codebook {
    std::vector<HUF_CElt> ctable;
    std::vector<std::uint8_t> serialized;
};

struct Huff0Decodebook {
    std::vector<HUF_DTable> dtable;
};

void check_huf(size_t code) {
    if (HUF_isError(code)) {
        throw std::runtime_error(HUF_getErrorName(code));
    }
}

PyObject* bytes_from_vector(const std::vector<std::uint8_t>& data) {
    const char* pointer = data.empty() ? "" : reinterpret_cast<const char*>(data.data());
    return PyBytes_FromStringAndSize(pointer, static_cast<Py_ssize_t>(data.size()));
}

std::array<unsigned, 256> count_symbols(const std::uint8_t* data, Py_ssize_t data_len, unsigned prior_count) {
    std::array<unsigned, 256> counts{};
    if (prior_count > 0) {
        counts.fill(prior_count);
    }
    for (Py_ssize_t index = 0; index < data_len; ++index) {
        if (counts[data[index]] == std::numeric_limits<unsigned>::max()) {
            throw std::runtime_error("Huff0 symbol count overflow.");
        }
        counts[data[index]] += 1;
    }
    return counts;
}

bool has_enough_symbols(const std::array<unsigned, 256>& counts) {
    int nonzero = 0;
    for (unsigned count : counts) {
        nonzero += count > 0 ? 1 : 0;
        if (nonzero >= 2) {
            return true;
        }
    }
    return false;
}

unsigned max_observed_symbol(const std::array<unsigned, 256>& counts) {
    for (int symbol = 255; symbol >= 0; --symbol) {
        if (counts[static_cast<std::size_t>(symbol)] > 0) {
            return static_cast<unsigned>(symbol);
        }
    }
    return 0;
}

std::unique_ptr<Huff0Codebook> make_codebook(
    const std::uint8_t* data,
    Py_ssize_t data_len,
    unsigned prior_count) {
    if (data_len <= 0) {
        throw std::runtime_error("Cannot build a Huff0 codebook from empty data.");
    }

    auto counts = count_symbols(data, data_len, prior_count);
    if (!has_enough_symbols(counts)) {
        throw std::runtime_error("Huff0 requires at least two symbols.");
    }
    unsigned const max_symbol_value = max_observed_symbol(counts);

    auto codebook = std::make_unique<Huff0Codebook>();
    codebook->ctable.resize(HUF_CTABLE_SIZE_ST(kMaxSymbolValue));

    std::vector<unsigned> build_workspace(HUF_CTABLE_WORKSPACE_SIZE_U32);
    size_t table_log = HUF_buildCTable_wksp(
        codebook->ctable.data(),
        counts.data(),
        max_symbol_value,
        kTableLog,
        build_workspace.data(),
        build_workspace.size() * sizeof(unsigned));
    check_huf(table_log);

    codebook->serialized.resize(HUF_CTABLEBOUND);
    std::vector<std::uint64_t> write_workspace(HUF_WORKSPACE_SIZE_U64);
    size_t header_size = HUF_writeCTable_wksp(
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

std::vector<std::uint8_t> encode_with_ctable(
    const std::uint8_t* data,
    Py_ssize_t data_len,
    const HUF_CElt* ctable,
    std::size_t max_output_bytes) {
    if (data_len <= 0 || static_cast<std::size_t>(data_len) > kMaxBlockSize) {
        return {};
    }
    std::size_t capacity = HUF_compressBound(static_cast<std::size_t>(data_len));
    if (max_output_bytes != std::numeric_limits<std::size_t>::max()) {
        capacity = std::min(capacity, max_output_bytes + 1);
    }
    if (capacity == 0) {
        return {};
    }

    std::vector<std::uint8_t> output(capacity);
    size_t size = HUF_compress1X_usingCTable(
        output.data(),
        output.size(),
        data,
        static_cast<std::size_t>(data_len),
        ctable,
        kFlags);
    if (HUF_isError(size)) {
        if (max_output_bytes != std::numeric_limits<std::size_t>::max()) {
            return {};
        }
        check_huf(size);
    }
    if (size == 0 || size > max_output_bytes) {
        return {};
    }
    output.resize(size);
    return output;
}

std::unique_ptr<Huff0Decodebook> make_decodebook_from_header(const std::uint8_t* header, Py_ssize_t header_len) {
    if (header_len <= 0) {
        throw std::runtime_error("Huff0 codebook header is empty.");
    }

    auto decodebook = std::make_unique<Huff0Decodebook>();
    decodebook->dtable.resize(HUF_DTABLE_SIZE(HUF_TABLELOG_MAX - 1));
    decodebook->dtable[0] = static_cast<HUF_DTable>((HUF_TABLELOG_MAX - 1) * 0x01000001U);
    std::vector<unsigned> workspace(HUF_DECOMPRESS_WORKSPACE_SIZE_U32);
    size_t read_size = HUF_readDTableX1_wksp(
        decodebook->dtable.data(),
        header,
        static_cast<std::size_t>(header_len),
        workspace.data(),
        workspace.size() * sizeof(unsigned),
        kFlags);
    check_huf(read_size);
    if (read_size != static_cast<std::size_t>(header_len)) {
        throw std::runtime_error("Huff0 did not consume the full codebook header.");
    }
    return decodebook;
}

std::vector<std::uint8_t> decode_with_dtable(
    const std::uint8_t* payload,
    Py_ssize_t payload_len,
    Py_ssize_t symbol_count,
    const HUF_DTable* dtable) {
    if (symbol_count < 0) {
        throw std::runtime_error("symbol_count must be non-negative.");
    }
    std::vector<std::uint8_t> output(static_cast<std::size_t>(symbol_count));
    if (symbol_count == 0) {
        return output;
    }
    size_t decoded = HUF_decompress1X_usingDTable(
        output.data(),
        output.size(),
        payload,
        static_cast<std::size_t>(payload_len),
        dtable,
        kFlags);
    check_huf(decoded);
    if (decoded != static_cast<std::size_t>(symbol_count)) {
        throw std::runtime_error("Huff0 decoded size mismatch.");
    }
    return output;
}

std::vector<std::uint8_t> restore_ngram_tokens(
    const std::vector<std::uint8_t>& tokens,
    int n,
    const std::vector<std::uint32_t>& vocab,
    const std::uint8_t* escapes,
    Py_ssize_t escape_len,
    const std::uint8_t* tail,
    Py_ssize_t tail_len) {
    if (n < 2 || n > 4) {
        throw std::runtime_error("n must be in [2, 4].");
    }
    if (tail_len < 0 || tail_len >= n) {
        throw std::runtime_error("Packed n-gram tail length is invalid.");
    }

    std::size_t escape_offset = 0;
    std::vector<std::uint8_t> output;
    output.reserve(tokens.size() * static_cast<std::size_t>(n)
                   + static_cast<std::size_t>(tail_len));
    for (std::uint8_t token : tokens) {
        if (token == 255) {
            std::size_t end = escape_offset + static_cast<std::size_t>(n);
            if (end > static_cast<std::size_t>(escape_len)) {
                throw std::runtime_error("Packed n-gram escape stream is truncated.");
            }
            output.insert(output.end(), escapes + escape_offset, escapes + end);
            escape_offset = end;
            continue;
        }
        if (static_cast<std::size_t>(token) >= vocab.size()) {
            throw std::runtime_error("Packed n-gram token is outside the vocabulary.");
        }
        std::uint32_t symbol = vocab[token];
        for (int offset = 0; offset < n; ++offset) {
            output.push_back(static_cast<std::uint8_t>((symbol >> (8 * offset)) & 0xFFU));
        }
    }
    if (escape_offset != static_cast<std::size_t>(escape_len)) {
        throw std::runtime_error("Packed n-gram escape stream contains unused bytes.");
    }
    output.insert(output.end(), tail, tail + tail_len);
    return output;
}

std::uint32_t pack_ngram_symbol(const std::uint8_t* data, int n) {
    std::uint32_t symbol = 0;
    for (int index = 0; index < n; ++index) {
        symbol |= static_cast<std::uint32_t>(data[index]) << (8 * index);
    }
    return symbol;
}

std::vector<std::uint32_t> vocab_from_view(const Py_buffer& vocab_view) {
    if (vocab_view.len % 4 != 0) {
        throw std::runtime_error("ngram vocab must contain little-endian uint32 values.");
    }
    const auto* data = static_cast<const std::uint8_t*>(vocab_view.buf);
    std::vector<std::uint32_t> vocab(static_cast<std::size_t>(vocab_view.len / 4));
    for (std::size_t index = 0; index < vocab.size(); ++index) {
        std::uint32_t value = 0;
        value |= static_cast<std::uint32_t>(data[index * 4]);
        value |= static_cast<std::uint32_t>(data[index * 4 + 1]) << 8;
        value |= static_cast<std::uint32_t>(data[index * 4 + 2]) << 16;
        value |= static_cast<std::uint32_t>(data[index * 4 + 3]) << 24;
        vocab[index] = value;
    }
    if (vocab.size() > 255) {
        throw std::runtime_error("ngram vocab must have at most 255 entries.");
    }
    if (!std::is_sorted(vocab.begin(), vocab.end())) {
        throw std::runtime_error("ngram vocab must be sorted.");
    }
    return vocab;
}

struct NGramTokenization {
    std::vector<std::uint8_t> tokens;
    std::vector<std::uint8_t> escapes;
    std::vector<std::uint8_t> tail;
    Py_ssize_t hits = 0;
    Py_ssize_t misses = 0;
};

NGramTokenization tokenize_ngrams(
    const std::uint8_t* data,
    Py_ssize_t data_len,
    int n,
    const std::vector<std::uint32_t>& vocab) {
    if (n < 1 || n > 4) {
        throw std::runtime_error("Huff0 n-gram tokenization supports n in [1, 4].");
    }
    if (vocab.empty()) {
        throw std::runtime_error("ngram vocab is empty.");
    }
    if (data_len < 0) {
        throw std::runtime_error("data length must be non-negative.");
    }

    Py_ssize_t complete_len = (data_len / n) * n;
    Py_ssize_t token_count = complete_len / n;
    NGramTokenization result;
    result.tokens.reserve(static_cast<std::size_t>(token_count));
    result.escapes.reserve(static_cast<std::size_t>(complete_len));
    if (data_len > complete_len) {
        result.tail.assign(data + complete_len, data + data_len);
    }

    for (Py_ssize_t token_index = 0; token_index < token_count; ++token_index) {
        const std::uint8_t* pointer = data + token_index * n;
        std::uint32_t symbol = pack_ngram_symbol(pointer, n);
        auto iterator = std::lower_bound(vocab.begin(), vocab.end(), symbol);
        if (iterator != vocab.end() && *iterator == symbol) {
            std::size_t position = static_cast<std::size_t>(iterator - vocab.begin());
            if (position >= 255) {
                throw std::runtime_error("ngram vocab id overflow.");
            }
            result.tokens.push_back(static_cast<std::uint8_t>(position));
            result.hits += 1;
            continue;
        }

        result.tokens.push_back(255);
        result.escapes.insert(result.escapes.end(), pointer, pointer + n);
        result.misses += 1;
    }
    return result;
}

void destroy_codebook_capsule(PyObject* capsule) {
    void* pointer = PyCapsule_GetPointer(capsule, "neuralzip.Huff0Codebook");
    delete static_cast<Huff0Codebook*>(pointer);
}

Huff0Codebook* codebook_from_capsule(PyObject* capsule) {
    void* pointer = PyCapsule_GetPointer(capsule, "neuralzip.Huff0Codebook");
    if (pointer == nullptr) {
        throw std::runtime_error("Invalid Huff0 codebook capsule.");
    }
    return static_cast<Huff0Codebook*>(pointer);
}

void destroy_decodebook_capsule(PyObject* capsule) {
    void* pointer = PyCapsule_GetPointer(capsule, "neuralzip.Huff0Decodebook");
    delete static_cast<Huff0Decodebook*>(pointer);
}

Huff0Decodebook* decodebook_from_capsule(PyObject* capsule) {
    void* pointer = PyCapsule_GetPointer(capsule, "neuralzip.Huff0Decodebook");
    if (pointer == nullptr) {
        throw std::runtime_error("Invalid Huff0 decodebook capsule.");
    }
    return static_cast<Huff0Decodebook*>(pointer);
}

PyObject* py_max_block_size(PyObject*, PyObject*) {
    return PyLong_FromSize_t(kMaxBlockSize);
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
    if (prior_count < 0 || prior_count > static_cast<Py_ssize_t>(std::numeric_limits<unsigned>::max())) {
        PyBuffer_Release(&data_view);
        PyErr_SetString(PyExc_ValueError, "prior_count must fit in unsigned.");
        return nullptr;
    }

    try {
        const auto* data = static_cast<const std::uint8_t*>(data_view.buf);
        auto codebook = make_codebook(data, data_view.len, static_cast<unsigned>(prior_count));
        PyObject* capsule = PyCapsule_New(
            codebook.get(),
            "neuralzip.Huff0Codebook",
            destroy_codebook_capsule);
        PyObject* header = bytes_from_vector(codebook->serialized);
        PyBuffer_Release(&data_view);

        if (!capsule || !header) {
            Py_XDECREF(capsule);
            Py_XDECREF(header);
            return nullptr;
        }
        codebook.release();

        PyObject* result = PyTuple_Pack(2, capsule, header);
        Py_DECREF(capsule);
        Py_DECREF(header);
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
    Py_buffer header_view;
    if (!PyArg_ParseTuple(args, "y*", &header_view)) {
        return nullptr;
    }
    try {
        const auto* header = static_cast<const std::uint8_t*>(header_view.buf);
        auto decodebook = make_decodebook_from_header(header, header_view.len);
        PyObject* capsule = PyCapsule_New(
            decodebook.get(),
            "neuralzip.Huff0Decodebook",
            destroy_decodebook_capsule);
        PyBuffer_Release(&header_view);
        if (!capsule) {
            return nullptr;
        }
        decodebook.release();
        return capsule;
    } catch (const std::bad_alloc&) {
        PyBuffer_Release(&header_view);
        PyErr_NoMemory();
        return nullptr;
    } catch (const std::exception& error) {
        PyBuffer_Release(&header_view);
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

PyObject* py_encode_with_codebook(PyObject*, PyObject* args, PyObject* kwargs) {
    Py_buffer data_view;
    PyObject* capsule = nullptr;
    Py_ssize_t min_saving_bytes = 0;
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
        Huff0Codebook* codebook = codebook_from_capsule(capsule);
        const auto* data = static_cast<const std::uint8_t*>(data_view.buf);
        std::size_t max_bytes =
            max_output_bytes < 0
                ? std::numeric_limits<std::size_t>::max()
                : static_cast<std::size_t>(max_output_bytes);
        std::vector<std::uint8_t> encoded =
            encode_with_ctable(data, data_view.len, codebook->ctable.data(), max_bytes);

        if (encoded.empty()
            || encoded.size() + static_cast<std::size_t>(min_saving_bytes)
                >= static_cast<std::size_t>(data_view.len)) {
            PyBuffer_Release(&data_view);
            Py_RETURN_NONE;
        }

        PyObject* payload = bytes_from_vector(encoded);
        PyObject* symbol_count = PyLong_FromSsize_t(data_view.len);
        PyBuffer_Release(&data_view);
        if (!payload || !symbol_count) {
            Py_XDECREF(payload);
            Py_XDECREF(symbol_count);
            return nullptr;
        }
        PyObject* result = PyTuple_Pack(2, payload, symbol_count);
        Py_DECREF(payload);
        Py_DECREF(symbol_count);
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

PyObject* py_encode_ngram_with_codebook(PyObject*, PyObject* args, PyObject* kwargs) {
    Py_buffer data_view;
    PyObject* capsule = nullptr;
    Py_buffer vocab_view;
    Py_ssize_t n = 0;
    Py_ssize_t min_saving_bytes = 0;
    Py_ssize_t max_output_bytes = -1;
    static const char* keyword_names[] = {
        "data",
        "codebook",
        "vocab",
        "n",
        "min_saving_bytes",
        "max_output_bytes",
        nullptr,
    };

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "y*Oy*n|nn",
            const_cast<char**>(keyword_names),
            &data_view,
            &capsule,
            &vocab_view,
            &n,
            &min_saving_bytes,
            &max_output_bytes)) {
        return nullptr;
    }
    if (n < 1 || n > 4) {
        PyBuffer_Release(&data_view);
        PyBuffer_Release(&vocab_view);
        PyErr_SetString(PyExc_ValueError, "n must be in [1, 4].");
        return nullptr;
    }
    if (min_saving_bytes < 0) {
        PyBuffer_Release(&data_view);
        PyBuffer_Release(&vocab_view);
        PyErr_SetString(PyExc_ValueError, "min_saving_bytes must be non-negative.");
        return nullptr;
    }

    try {
        Huff0Codebook* codebook = codebook_from_capsule(capsule);
        const auto* data = static_cast<const std::uint8_t*>(data_view.buf);
        std::vector<std::uint32_t> vocab = vocab_from_view(vocab_view);
        NGramTokenization tokens =
            tokenize_ngrams(data, data_view.len, static_cast<int>(n), vocab);

        if (tokens.tokens.empty()) {
            PyBuffer_Release(&data_view);
            PyBuffer_Release(&vocab_view);
            Py_RETURN_NONE;
        }

        std::size_t max_bytes =
            max_output_bytes < 0
                ? std::numeric_limits<std::size_t>::max()
                : static_cast<std::size_t>(max_output_bytes);
        std::vector<std::uint8_t> encoded =
            encode_with_ctable(
                tokens.tokens.data(),
                static_cast<Py_ssize_t>(tokens.tokens.size()),
                codebook->ctable.data(),
                max_bytes);

        if (encoded.empty()
            || encoded.size() + static_cast<std::size_t>(min_saving_bytes)
                >= tokens.tokens.size()) {
            PyBuffer_Release(&data_view);
            PyBuffer_Release(&vocab_view);
            Py_RETURN_NONE;
        }

        PyObject* payload = bytes_from_vector(encoded);
        PyObject* symbol_count = PyLong_FromSize_t(tokens.tokens.size());
        PyObject* escapes = bytes_from_vector(tokens.escapes);
        PyObject* tail = bytes_from_vector(tokens.tail);
        PyObject* hits = PyLong_FromSsize_t(tokens.hits);
        PyObject* misses = PyLong_FromSsize_t(tokens.misses);
        PyBuffer_Release(&data_view);
        PyBuffer_Release(&vocab_view);
        if (!payload || !symbol_count || !escapes || !tail || !hits || !misses) {
            Py_XDECREF(payload);
            Py_XDECREF(symbol_count);
            Py_XDECREF(escapes);
            Py_XDECREF(tail);
            Py_XDECREF(hits);
            Py_XDECREF(misses);
            return nullptr;
        }
        PyObject* result = PyTuple_Pack(6, payload, symbol_count, escapes, tail, hits, misses);
        Py_DECREF(payload);
        Py_DECREF(symbol_count);
        Py_DECREF(escapes);
        Py_DECREF(tail);
        Py_DECREF(hits);
        Py_DECREF(misses);
        return result;
    } catch (const std::bad_alloc&) {
        PyBuffer_Release(&data_view);
        PyBuffer_Release(&vocab_view);
        PyErr_NoMemory();
        return nullptr;
    } catch (const std::exception& error) {
        PyBuffer_Release(&data_view);
        PyBuffer_Release(&vocab_view);
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

PyObject* py_compress(PyObject*, PyObject* args, PyObject* kwargs) {
    Py_buffer data_view;
    Py_ssize_t min_saving_bytes = 0;
    Py_ssize_t prior_count = 0;
    static const char* keyword_names[] = {"data", "min_saving_bytes", "prior_count", nullptr};

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "y*|nn",
            const_cast<char**>(keyword_names),
            &data_view,
            &min_saving_bytes,
            &prior_count)) {
        return nullptr;
    }
    if (min_saving_bytes < 0 || prior_count < 0) {
        PyBuffer_Release(&data_view);
        PyErr_SetString(PyExc_ValueError, "min_saving_bytes and prior_count must be non-negative.");
        return nullptr;
    }

    try {
        const auto* data = static_cast<const std::uint8_t*>(data_view.buf);
        auto codebook = make_codebook(data, data_view.len, static_cast<unsigned>(prior_count));
        std::vector<std::uint8_t> encoded =
            encode_with_ctable(data, data_view.len, codebook->ctable.data(), std::numeric_limits<std::size_t>::max());
        if (encoded.empty()
            || encoded.size() + codebook->serialized.size() + static_cast<std::size_t>(min_saving_bytes)
                >= static_cast<std::size_t>(data_view.len)) {
            PyBuffer_Release(&data_view);
            Py_RETURN_NONE;
        }

        PyObject* payload = bytes_from_vector(encoded);
        PyObject* header = bytes_from_vector(codebook->serialized);
        PyObject* symbol_count = PyLong_FromSsize_t(data_view.len);
        PyBuffer_Release(&data_view);
        if (!payload || !header || !symbol_count) {
            Py_XDECREF(payload);
            Py_XDECREF(header);
            Py_XDECREF(symbol_count);
            return nullptr;
        }
        PyObject* result = PyTuple_Pack(3, payload, header, symbol_count);
        Py_DECREF(payload);
        Py_DECREF(header);
        Py_DECREF(symbol_count);
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
    Py_buffer header_view;
    Py_ssize_t symbol_count;
    if (!PyArg_ParseTuple(args, "y*y*n", &payload_view, &header_view, &symbol_count)) {
        return nullptr;
    }

    try {
        const auto* header = static_cast<const std::uint8_t*>(header_view.buf);
        auto decodebook = make_decodebook_from_header(header, header_view.len);
        const auto* payload = static_cast<const std::uint8_t*>(payload_view.buf);
        std::vector<std::uint8_t> decoded =
            decode_with_dtable(payload, payload_view.len, symbol_count, decodebook->dtable.data());
        PyObject* result = bytes_from_vector(decoded);
        PyBuffer_Release(&payload_view);
        PyBuffer_Release(&header_view);
        return result;
    } catch (const std::bad_alloc&) {
        PyBuffer_Release(&payload_view);
        PyBuffer_Release(&header_view);
        PyErr_NoMemory();
        return nullptr;
    } catch (const std::exception& error) {
        PyBuffer_Release(&payload_view);
        PyBuffer_Release(&header_view);
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

    try {
        Huff0Decodebook* decodebook = decodebook_from_capsule(capsule);
        const auto* payload = static_cast<const std::uint8_t*>(payload_view.buf);
        std::vector<std::uint8_t> decoded =
            decode_with_dtable(payload, payload_view.len, symbol_count, decodebook->dtable.data());
        PyObject* result = bytes_from_vector(decoded);
        PyBuffer_Release(&payload_view);
        return result;
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

PyObject* py_decompress_ngram_with_decodebook(PyObject*, PyObject* args) {
    Py_buffer payload_view{};
    PyObject* capsule = nullptr;
    Py_ssize_t symbol_count = 0;
    Py_buffer vocab_view{};
    Py_ssize_t n = 0;
    Py_buffer escape_view{};
    Py_buffer tail_view{};
    if (!PyArg_ParseTuple(
            args,
            "y*Ony*ny*y*",
            &payload_view,
            &capsule,
            &symbol_count,
            &vocab_view,
            &n,
            &escape_view,
            &tail_view)) {
        return nullptr;
    }

    auto release_buffers = [&]() {
        PyBuffer_Release(&payload_view);
        PyBuffer_Release(&vocab_view);
        PyBuffer_Release(&escape_view);
        PyBuffer_Release(&tail_view);
    };

    try {
        Huff0Decodebook* decodebook = decodebook_from_capsule(capsule);
        const auto* payload = static_cast<const std::uint8_t*>(payload_view.buf);
        std::vector<std::uint8_t> tokens = decode_with_dtable(
            payload,
            payload_view.len,
            symbol_count,
            decodebook->dtable.data());
        std::vector<std::uint32_t> vocab = vocab_from_view(vocab_view);
        std::vector<std::uint8_t> decoded = restore_ngram_tokens(
            tokens,
            static_cast<int>(n),
            vocab,
            static_cast<const std::uint8_t*>(escape_view.buf),
            escape_view.len,
            static_cast<const std::uint8_t*>(tail_view.buf),
            tail_view.len);
        PyObject* result = bytes_from_vector(decoded);
        release_buffers();
        return result;
    } catch (const std::bad_alloc&) {
        release_buffers();
        PyErr_NoMemory();
        return nullptr;
    } catch (const std::exception& error) {
        release_buffers();
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

static PyMethodDef methods[] = {
    {"max_block_size", py_max_block_size, METH_NOARGS, "Return Huff0 maximum block size."},
    {"compress", reinterpret_cast<PyCFunction>(py_compress), METH_VARARGS | METH_KEYWORDS, "Compress one block with Huff0."},
    {"build_codebook", reinterpret_cast<PyCFunction>(py_build_codebook), METH_VARARGS | METH_KEYWORDS, "Build a Huff0 codebook."},
    {"build_decodebook", py_build_decodebook, METH_VARARGS, "Build a Huff0 decode table from a serialized codebook."},
    {"encode_with_codebook", reinterpret_cast<PyCFunction>(py_encode_with_codebook), METH_VARARGS | METH_KEYWORDS, "Encode using a Huff0 codebook."},
    {"encode_ngram_with_codebook", reinterpret_cast<PyCFunction>(py_encode_ngram_with_codebook), METH_VARARGS | METH_KEYWORDS, "Tokenize n-grams and encode token ids using a Huff0 codebook."},
    {"decompress", py_decompress, METH_VARARGS, "Decompress using a serialized Huff0 codebook."},
    {"decompress_with_decodebook", py_decompress_with_decodebook, METH_VARARGS, "Decompress using a Huff0 decodebook."},
    {"decompress_ngram_with_decodebook", py_decompress_ngram_with_decodebook, METH_VARARGS, "Decode Huff0 token ids and restore packed n-grams."},
    {nullptr, nullptr, 0, nullptr},
};

static PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_huff0_cpp",
    "Direct Huff0 wrapper backed by vendored Zstandard HUF APIs.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__huff0_cpp() {
    return PyModule_Create(&module);
}
