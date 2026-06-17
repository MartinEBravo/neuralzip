#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct SparseArithmeticCodebook {
    int n = 0;
    std::vector<std::uint32_t> symbols;
    std::vector<std::uint32_t> frequencies;
    std::vector<std::uint32_t> cumulative;
    std::unordered_map<std::uint32_t, std::size_t> index_by_symbol;
};

class ArithmeticBitWriter {
public:
    void write_bit(int bit) {
        current_ = static_cast<std::uint8_t>((current_ << 1) | (bit & 1));
        bit_count_ += 1;
        if (bit_count_ == 8) {
            output_.push_back(current_);
            current_ = 0;
            bit_count_ = 0;
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

class ArithmeticBitReader {
public:
    ArithmeticBitReader(const std::uint8_t* data, Py_ssize_t length)
        : data_(data), length_(length) {}

    int read_bit() {
        if (byte_index_ >= length_) {
            return 0;
        }
        int bit = (data_[byte_index_] >> bit_index_) & 1;
        bit_index_ -= 1;
        if (bit_index_ < 0) {
            bit_index_ = 7;
            byte_index_ += 1;
        }
        return bit;
    }

private:
    const std::uint8_t* data_;
    Py_ssize_t length_;
    Py_ssize_t byte_index_ = 0;
    int bit_index_ = 7;
};

PyObject* bytes_from_vector(const std::vector<std::uint8_t>& data) {
    const char* pointer = data.empty() ? "" : reinterpret_cast<const char*>(data.data());
    return PyBytes_FromStringAndSize(pointer, static_cast<Py_ssize_t>(data.size()));
}

void append_u32_le(std::vector<std::uint8_t>& output, std::uint32_t value) {
    output.push_back(static_cast<std::uint8_t>(value & 0xFF));
    output.push_back(static_cast<std::uint8_t>((value >> 8) & 0xFF));
    output.push_back(static_cast<std::uint8_t>((value >> 16) & 0xFF));
    output.push_back(static_cast<std::uint8_t>((value >> 24) & 0xFF));
}

std::uint32_t read_u32_le(const std::uint8_t* data) {
    std::uint32_t value = 0;
    value |= static_cast<std::uint32_t>(data[0]);
    value |= static_cast<std::uint32_t>(data[1]) << 8;
    value |= static_cast<std::uint32_t>(data[2]) << 16;
    value |= static_cast<std::uint32_t>(data[3]) << 24;
    return value;
}

std::uint32_t pack_ngram_symbol(const std::uint8_t* data, int n) {
    std::uint32_t symbol = 0;
    for (int index = 0; index < n; ++index) {
        symbol |= static_cast<std::uint32_t>(data[index]) << (8 * index);
    }
    return symbol;
}

void append_ngram_symbol(std::vector<std::uint8_t>& output, std::uint32_t symbol, int n) {
    for (int index = 0; index < n; ++index) {
        output.push_back(static_cast<std::uint8_t>((symbol >> (8 * index)) & 0xFF));
    }
}

void finalize_sparse_codebook(SparseArithmeticCodebook& codebook) {
    codebook.cumulative.assign(codebook.frequencies.size() + 1, 0);
    codebook.index_by_symbol.clear();
    for (std::size_t index = 0; index < codebook.frequencies.size(); ++index) {
        if (codebook.frequencies[index] == 0) {
            throw std::runtime_error("Sparse arithmetic frequencies must be positive.");
        }
        codebook.cumulative[index + 1] = codebook.cumulative[index] + codebook.frequencies[index];
        codebook.index_by_symbol[codebook.symbols[index]] = index;
    }
    if (codebook.cumulative.empty() || codebook.cumulative.back() == 0) {
        throw std::runtime_error("Sparse arithmetic codebook is empty.");
    }
}

SparseArithmeticCodebook make_sparse_ngram_codebook(
    const std::uint8_t* data,
    Py_ssize_t data_len,
    int n,
    std::uint32_t max_total = 1U << 20) {
    if (n < 1 || n > 4) {
        throw std::runtime_error("Sparse n-gram arithmetic supports n in [1, 4].");
    }

    Py_ssize_t token_count = data_len / n;
    if (token_count <= 0) {
        throw std::runtime_error("Cannot build a sparse n-gram codebook from an empty token stream.");
    }

    std::unordered_map<std::uint32_t, std::uint64_t> counts;
    counts.reserve(static_cast<std::size_t>(std::min<Py_ssize_t>(token_count, 65536)));
    for (Py_ssize_t index = 0; index < token_count; ++index) {
        std::uint32_t symbol = pack_ngram_symbol(data + index * n, n);
        counts[symbol] += 1;
    }

    if (counts.size() > max_total) {
        throw std::runtime_error("Sparse n-gram alphabet is larger than max_total.");
    }

    std::vector<std::pair<std::uint32_t, std::uint64_t>> ordered;
    ordered.reserve(counts.size());
    std::uint64_t total = 0;
    for (const auto& entry : counts) {
        ordered.push_back(entry);
        total += entry.second;
    }
    std::sort(ordered.begin(), ordered.end());

    SparseArithmeticCodebook codebook;
    codebook.n = n;
    codebook.symbols.reserve(ordered.size());
    codebook.frequencies.reserve(ordered.size());

    if (total <= max_total) {
        for (const auto& [symbol, count] : ordered) {
            codebook.symbols.push_back(symbol);
            codebook.frequencies.push_back(static_cast<std::uint32_t>(count));
        }
        finalize_sparse_codebook(codebook);
        return codebook;
    }

    double scale = static_cast<double>(max_total) / static_cast<double>(total);
    std::uint32_t frequency_total = 0;
    std::vector<std::uint64_t> raw_counts;
    raw_counts.reserve(ordered.size());
    for (const auto& [symbol, count] : ordered) {
        std::uint32_t frequency = std::max<std::uint32_t>(
            1,
            static_cast<std::uint32_t>(static_cast<double>(count) * scale));
        codebook.symbols.push_back(symbol);
        codebook.frequencies.push_back(frequency);
        raw_counts.push_back(count);
        frequency_total += frequency;
    }

    while (frequency_total > max_total) {
        std::size_t best_index = codebook.frequencies.size();
        for (std::size_t index = 0; index < codebook.frequencies.size(); ++index) {
            if (codebook.frequencies[index] <= 1) {
                continue;
            }
            if (best_index == codebook.frequencies.size()
                || codebook.frequencies[index] > codebook.frequencies[best_index]) {
                best_index = index;
            }
        }
        if (best_index == codebook.frequencies.size()) {
            break;
        }
        codebook.frequencies[best_index] -= 1;
        frequency_total -= 1;
    }

    while (frequency_total < max_total) {
        std::size_t best_index = 0;
        for (std::size_t index = 1; index < raw_counts.size(); ++index) {
            if (raw_counts[index] > raw_counts[best_index]) {
                best_index = index;
            }
        }
        codebook.frequencies[best_index] += 1;
        frequency_total += 1;
    }

    finalize_sparse_codebook(codebook);
    return codebook;
}

std::vector<std::uint8_t> sparse_codebook_to_bytes(const SparseArithmeticCodebook& codebook) {
    if (codebook.n < 1 || codebook.n > 4) {
        throw std::runtime_error("Invalid sparse n-gram size.");
    }
    if (codebook.symbols.size() != codebook.frequencies.size()) {
        throw std::runtime_error("Sparse codebook has inconsistent arrays.");
    }
    if (codebook.symbols.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("Sparse codebook is too large to serialize.");
    }

    std::vector<std::uint8_t> output;
    output.reserve(5 + codebook.symbols.size() * 8);
    output.push_back(static_cast<std::uint8_t>(codebook.n));
    append_u32_le(output, static_cast<std::uint32_t>(codebook.symbols.size()));
    for (std::size_t index = 0; index < codebook.symbols.size(); ++index) {
        append_u32_le(output, codebook.symbols[index]);
        append_u32_le(output, codebook.frequencies[index]);
    }
    return output;
}

SparseArithmeticCodebook sparse_codebook_from_view(const Py_buffer& codebook_view) {
    if (codebook_view.len < 5) {
        throw std::runtime_error("Sparse codebook payload is too short.");
    }
    const auto* data = static_cast<const std::uint8_t*>(codebook_view.buf);
    int n = static_cast<int>(data[0]);
    if (n < 1 || n > 4) {
        throw std::runtime_error("Sparse n-gram arithmetic supports n in [1, 4].");
    }
    std::uint32_t size = read_u32_le(data + 1);
    std::uint64_t expected = 5ULL + static_cast<std::uint64_t>(size) * 8ULL;
    if (expected != static_cast<std::uint64_t>(codebook_view.len)) {
        throw std::runtime_error("Sparse codebook payload has invalid length.");
    }

    SparseArithmeticCodebook codebook;
    codebook.n = n;
    codebook.symbols.reserve(size);
    codebook.frequencies.reserve(size);
    const std::uint8_t* cursor = data + 5;
    for (std::uint32_t index = 0; index < size; ++index) {
        std::uint32_t symbol = read_u32_le(cursor);
        std::uint32_t frequency = read_u32_le(cursor + 4);
        codebook.symbols.push_back(symbol);
        codebook.frequencies.push_back(frequency);
        cursor += 8;
    }
    finalize_sparse_codebook(codebook);
    return codebook;
}

std::vector<std::uint8_t> sparse_arithmetic_encode_ngrams(
    const std::uint8_t* data,
    Py_ssize_t data_len,
    const SparseArithmeticCodebook& codebook,
    std::size_t max_output_bytes = std::numeric_limits<std::size_t>::max()) {
    constexpr std::uint64_t TOP = 0xFFFFFFFFULL;
    constexpr std::uint64_t HALF = 0x80000000ULL;
    constexpr std::uint64_t FIRST_QTR = 0x40000000ULL;
    constexpr std::uint64_t THIRD_QTR = 0xC0000000ULL;

    Py_ssize_t token_count = data_len / codebook.n;
    std::uint64_t total = codebook.cumulative.back();
    std::uint64_t low = 0;
    std::uint64_t high = TOP;
    std::uint64_t pending = 0;
    ArithmeticBitWriter writer;

    auto write_bit_plus_pending = [&](int bit) {
        writer.write_bit(bit);
        while (pending > 0) {
            writer.write_bit(bit ? 0 : 1);
            pending -= 1;
        }
    };

    for (Py_ssize_t index = 0; index < token_count; ++index) {
        std::uint32_t symbol = pack_ngram_symbol(data + index * codebook.n, codebook.n);
        auto found = codebook.index_by_symbol.find(symbol);
        if (found == codebook.index_by_symbol.end()) {
            throw std::runtime_error("Input contains an n-gram missing from sparse codebook.");
        }
        std::size_t symbol_index = found->second;

        std::uint64_t range = high - low + 1;
        high = low + (range * codebook.cumulative[symbol_index + 1]) / total - 1;
        low = low + (range * codebook.cumulative[symbol_index]) / total;

        while (true) {
            if (high < HALF) {
                write_bit_plus_pending(0);
            } else if (low >= HALF) {
                write_bit_plus_pending(1);
                low -= HALF;
                high -= HALF;
            } else if (low >= FIRST_QTR && high < THIRD_QTR) {
                pending += 1;
                low -= FIRST_QTR;
                high -= FIRST_QTR;
            } else {
                break;
            }
            low = (low << 1) & TOP;
            high = ((high << 1) & TOP) | 1;
            if (writer.exceeds(max_output_bytes)) {
                return {};
            }
        }
    }

    pending += 1;
    if (low < FIRST_QTR) {
        write_bit_plus_pending(0);
    } else {
        write_bit_plus_pending(1);
    }
    if (writer.exceeds(max_output_bytes)) {
        return {};
    }
    return writer.finish();
}

std::vector<std::uint8_t> sparse_arithmetic_decode_ngrams(
    const std::uint8_t* payload,
    Py_ssize_t payload_len,
    const SparseArithmeticCodebook& codebook,
    Py_ssize_t symbol_count) {
    constexpr std::uint64_t TOP = 0xFFFFFFFFULL;
    constexpr std::uint64_t HALF = 0x80000000ULL;
    constexpr std::uint64_t FIRST_QTR = 0x40000000ULL;
    constexpr std::uint64_t THIRD_QTR = 0xC0000000ULL;

    std::uint64_t total = codebook.cumulative.back();
    std::uint64_t low = 0;
    std::uint64_t high = TOP;
    std::uint64_t code = 0;
    ArithmeticBitReader reader(payload, payload_len);
    for (int index = 0; index < 32; ++index) {
        code = ((code << 1) | static_cast<std::uint64_t>(reader.read_bit())) & TOP;
    }

    std::vector<std::uint8_t> output;
    output.reserve(static_cast<std::size_t>(symbol_count) * static_cast<std::size_t>(codebook.n));
    for (Py_ssize_t index = 0; index < symbol_count; ++index) {
        std::uint64_t range = high - low + 1;
        std::uint64_t scaled = (((code - low + 1) * total) - 1) / range;

        auto upper = std::upper_bound(codebook.cumulative.begin(), codebook.cumulative.end(), scaled);
        if (upper == codebook.cumulative.begin()) {
            throw std::runtime_error("Invalid sparse arithmetic payload.");
        }
        std::size_t symbol_index = static_cast<std::size_t>(upper - codebook.cumulative.begin() - 1);
        if (symbol_index >= codebook.symbols.size()) {
            throw std::runtime_error("Invalid sparse arithmetic payload.");
        }
        append_ngram_symbol(output, codebook.symbols[symbol_index], codebook.n);

        high = low + (range * codebook.cumulative[symbol_index + 1]) / total - 1;
        low = low + (range * codebook.cumulative[symbol_index]) / total;

        while (true) {
            if (high < HALF) {
                // no-op
            } else if (low >= HALF) {
                code -= HALF;
                low -= HALF;
                high -= HALF;
            } else if (low >= FIRST_QTR && high < THIRD_QTR) {
                code -= FIRST_QTR;
                low -= FIRST_QTR;
                high -= FIRST_QTR;
            } else {
                break;
            }
            low = (low << 1) & TOP;
            high = ((high << 1) & TOP) | 1;
            code = ((code << 1) & TOP) | static_cast<std::uint64_t>(reader.read_bit());
        }
    }
    return output;
}

PyObject* py_sparse_ngram_build_codebook(PyObject*, PyObject* args, PyObject* kwargs) {
    Py_buffer data_view;
    int n;
    Py_ssize_t max_total = 1 << 20;
    static const char* keyword_names[] = {"data", "n", "max_total", nullptr};

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "y*i|n",
            const_cast<char**>(keyword_names),
            &data_view,
            &n,
            &max_total)) {
        return nullptr;
    }

    if (max_total <= 0 || max_total > static_cast<Py_ssize_t>(std::numeric_limits<std::uint32_t>::max())) {
        PyBuffer_Release(&data_view);
        PyErr_SetString(PyExc_ValueError, "max_total must fit in uint32 and be positive.");
        return nullptr;
    }

    try {
        const auto* data = static_cast<const std::uint8_t*>(data_view.buf);
        SparseArithmeticCodebook codebook = make_sparse_ngram_codebook(
            data,
            data_view.len,
            n,
            static_cast<std::uint32_t>(max_total));
        std::vector<std::uint8_t> bytes = sparse_codebook_to_bytes(codebook);
        PyBuffer_Release(&data_view);
        return bytes_from_vector(bytes);
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

PyObject* py_sparse_ngram_encode_with_codebook(
    PyObject*,
    PyObject* args,
    PyObject* kwargs) {
    Py_buffer data_view;
    Py_buffer codebook_view;
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
            "y*y*|nn",
            const_cast<char**>(keyword_names),
            &data_view,
            &codebook_view,
            &min_saving_bytes,
            &max_output_bytes)) {
        return nullptr;
    }

    if (min_saving_bytes < 0) {
        PyBuffer_Release(&data_view);
        PyBuffer_Release(&codebook_view);
        PyErr_SetString(PyExc_ValueError, "min_saving_bytes must be non-negative.");
        return nullptr;
    }

    try {
        SparseArithmeticCodebook codebook = sparse_codebook_from_view(codebook_view);
        Py_ssize_t symbol_count = data_view.len / codebook.n;
        Py_ssize_t tail_size = data_view.len - symbol_count * codebook.n;
        if (symbol_count == 0) {
            PyBuffer_Release(&data_view);
            PyBuffer_Release(&codebook_view);
            Py_RETURN_NONE;
        }

        const auto* data = static_cast<const std::uint8_t*>(data_view.buf);
        std::size_t max_bytes =
            max_output_bytes < 0
                ? std::numeric_limits<std::size_t>::max()
                : static_cast<std::size_t>(max_output_bytes);

        std::vector<std::uint8_t> encoded =
            sparse_arithmetic_encode_ngrams(data, data_view.len, codebook, max_bytes);
        if (encoded.empty() && symbol_count > 0) {
            PyBuffer_Release(&data_view);
            PyBuffer_Release(&codebook_view);
            Py_RETURN_NONE;
        }
        if (encoded.size() + static_cast<std::size_t>(tail_size)
                + static_cast<std::size_t>(min_saving_bytes)
            >= static_cast<std::size_t>(data_view.len)) {
            PyBuffer_Release(&data_view);
            PyBuffer_Release(&codebook_view);
            Py_RETURN_NONE;
        }

        PyObject* payload_object = bytes_from_vector(encoded);
        PyObject* count_object = PyLong_FromSsize_t(symbol_count);
        PyObject* tail_object = PyBytes_FromStringAndSize(
            reinterpret_cast<const char*>(data + symbol_count * codebook.n),
            tail_size);
        PyBuffer_Release(&data_view);
        PyBuffer_Release(&codebook_view);

        if (!payload_object || !count_object || !tail_object) {
            Py_XDECREF(payload_object);
            Py_XDECREF(count_object);
            Py_XDECREF(tail_object);
            return nullptr;
        }

        PyObject* result = PyTuple_Pack(3, payload_object, count_object, tail_object);
        Py_DECREF(payload_object);
        Py_DECREF(count_object);
        Py_DECREF(tail_object);
        return result;
    } catch (const std::bad_alloc&) {
        PyBuffer_Release(&data_view);
        PyBuffer_Release(&codebook_view);
        PyErr_NoMemory();
        return nullptr;
    } catch (const std::exception& error) {
        PyBuffer_Release(&data_view);
        PyBuffer_Release(&codebook_view);
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

PyObject* py_sparse_ngram_decompress(PyObject*, PyObject* args) {
    Py_buffer payload_view;
    Py_buffer codebook_view;
    Py_buffer tail_view;
    Py_ssize_t symbol_count;

    if (!PyArg_ParseTuple(args, "y*y*ny*", &payload_view, &codebook_view, &symbol_count, &tail_view)) {
        return nullptr;
    }

    if (symbol_count < 0) {
        PyBuffer_Release(&payload_view);
        PyBuffer_Release(&codebook_view);
        PyBuffer_Release(&tail_view);
        PyErr_SetString(PyExc_ValueError, "symbol_count must be non-negative.");
        return nullptr;
    }

    try {
        SparseArithmeticCodebook codebook = sparse_codebook_from_view(codebook_view);
        if (tail_view.len >= codebook.n) {
            PyBuffer_Release(&payload_view);
            PyBuffer_Release(&codebook_view);
            PyBuffer_Release(&tail_view);
            PyErr_SetString(PyExc_ValueError, "tail length must be smaller than n.");
            return nullptr;
        }

        const auto* payload = static_cast<const std::uint8_t*>(payload_view.buf);
        std::vector<std::uint8_t> output = sparse_arithmetic_decode_ngrams(
            payload,
            payload_view.len,
            codebook,
            symbol_count);
        const auto* tail = static_cast<const std::uint8_t*>(tail_view.buf);
        output.insert(output.end(), tail, tail + tail_view.len);
        PyBuffer_Release(&payload_view);
        PyBuffer_Release(&codebook_view);
        PyBuffer_Release(&tail_view);
        return bytes_from_vector(output);
    } catch (const std::bad_alloc&) {
        PyBuffer_Release(&payload_view);
        PyBuffer_Release(&codebook_view);
        PyBuffer_Release(&tail_view);
        PyErr_NoMemory();
        return nullptr;
    } catch (const std::exception& error) {
        PyBuffer_Release(&payload_view);
        PyBuffer_Release(&codebook_view);
        PyBuffer_Release(&tail_view);
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

PyMethodDef methods[] = {
    {"sparse_ngram_build_codebook", reinterpret_cast<PyCFunction>(py_sparse_ngram_build_codebook), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"sparse_ngram_encode_with_codebook", reinterpret_cast<PyCFunction>(py_sparse_ngram_encode_with_codebook), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"sparse_ngram_decompress", py_sparse_ngram_decompress, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_arithmetic_cpp",
    nullptr,
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__arithmetic_cpp() {
    return PyModule_Create(&module);
}
