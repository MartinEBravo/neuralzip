#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr uint32_t CODE_BITS = 32;
constexpr uint64_t TOP = (uint64_t{1} << CODE_BITS) - 1;
constexpr uint64_t HALF = uint64_t{1} << (CODE_BITS - 1);
constexpr uint64_t FIRST_QTR = HALF >> 1;
constexpr uint64_t THIRD_QTR = FIRST_QTR * 3;

struct BitWriter {
    std::vector<uint8_t> bytes;
    uint8_t current = 0;
    int bits_filled = 0;

    void write_bit(uint32_t bit) {
        current = static_cast<uint8_t>((current << 1) | (bit & 1U));
        bits_filled += 1;
        if (bits_filled == 8) {
            bytes.push_back(current);
            current = 0;
            bits_filled = 0;
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

}  // namespace

extern "C" int arithmetic_encode(
    const uint32_t* symbols,
    size_t symbol_count,
    const uint32_t* counts,
    size_t count_count,
    uint8_t** out_bytes,
    size_t* out_size
) {
    if (!out_bytes || !out_size || (!symbols && symbol_count) || (!counts && count_count)) {
        return 1;
    }

    std::vector<uint64_t> cumulative(count_count + 1, 0);
    for (size_t i = 0; i < count_count; ++i) {
        cumulative[i + 1] = cumulative[i] + counts[i];
    }
    const uint64_t total = cumulative.back();
    if (total == 0 && symbol_count != 0) {
        return 2;
    }

    uint64_t low = 0;
    uint64_t high = TOP;
    uint32_t pending = 0;
    BitWriter writer;

    auto emit = [&](uint32_t bit) {
        writer.write_bit(bit);
        while (pending > 0) {
            writer.write_bit(1U - bit);
            pending -= 1;
        }
    };

    for (size_t i = 0; i < symbol_count; ++i) {
        uint32_t symbol = symbols[i];
        if (symbol + 1 >= cumulative.size()) {
            return 3;
        }
        uint64_t range = high - low + 1;
        high = low + (range * cumulative[symbol + 1]) / total - 1;
        low = low + (range * cumulative[symbol]) / total;

        while (true) {
            if (high < HALF) {
                emit(0);
            } else if (low >= HALF) {
                emit(1);
                low -= HALF;
                high -= HALF;
            } else if (low >= FIRST_QTR && high < THIRD_QTR) {
                pending += 1;
                low -= FIRST_QTR;
                high -= FIRST_QTR;
            } else {
                break;
            }
            low <<= 1;
            high = (high << 1) + 1;
        }
    }

    pending += 1;
    if (low < FIRST_QTR) {
        emit(0);
    } else {
        emit(1);
    }

    std::vector<uint8_t> encoded = writer.finish();
    *out_size = encoded.size();
    *out_bytes = static_cast<uint8_t*>(std::malloc(*out_size));
    if (!*out_bytes && *out_size != 0) {
        return 4;
    }
    if (*out_size != 0) {
        std::memcpy(*out_bytes, encoded.data(), *out_size);
    }
    return 0;
}

extern "C" int arithmetic_decode(
    const uint8_t* data,
    size_t data_size,
    const uint32_t* counts,
    size_t count_count,
    size_t symbol_count,
    uint32_t** out_symbols
) {
    if (!out_symbols || (!data && data_size) || (!counts && count_count)) {
        return 1;
    }

    std::vector<uint64_t> cumulative(count_count + 1, 0);
    for (size_t i = 0; i < count_count; ++i) {
        cumulative[i + 1] = cumulative[i] + counts[i];
    }
    const uint64_t total = cumulative.back();
    if (total == 0 && symbol_count != 0) {
        return 2;
    }

    *out_symbols = static_cast<uint32_t*>(std::malloc(symbol_count * sizeof(uint32_t)));
    if (!*out_symbols && symbol_count != 0) {
        return 3;
    }

    BitReader reader{data, data_size};
    uint64_t low = 0;
    uint64_t high = TOP;
    uint64_t code = 0;
    for (uint32_t i = 0; i < CODE_BITS; ++i) {
        code = (code << 1) | reader.read_bit();
    }

    for (size_t idx = 0; idx < symbol_count; ++idx) {
        uint64_t range = high - low + 1;
        uint64_t value = ((code - low + 1) * total - 1) / range;

        size_t symbol = 0;
        size_t left = 0;
        size_t right = count_count;
        while (left < right) {
            size_t mid = left + (right - left) / 2;
            if (cumulative[mid + 1] <= value) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        symbol = left;
        if (symbol >= count_count) {
            std::free(*out_symbols);
            *out_symbols = nullptr;
            return 4;
        }

        (*out_symbols)[idx] = static_cast<uint32_t>(symbol);
        high = low + (range * cumulative[symbol + 1]) / total - 1;
        low = low + (range * cumulative[symbol]) / total;

        while (true) {
            if (high < HALF) {
            } else if (low >= HALF) {
                low -= HALF;
                high -= HALF;
                code -= HALF;
            } else if (low >= FIRST_QTR && high < THIRD_QTR) {
                low -= FIRST_QTR;
                high -= FIRST_QTR;
                code -= FIRST_QTR;
            } else {
                break;
            }
            low <<= 1;
            high = (high << 1) + 1;
            code = (code << 1) + reader.read_bit();
        }
    }

    return 0;
}

extern "C" void arithmetic_free(void* ptr) {
    std::free(ptr);
}
