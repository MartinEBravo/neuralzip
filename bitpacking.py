import numpy as np


def bits_required(value_count):
    if value_count <= 1:
        return 0
    return (value_count - 1).bit_length()


def packed_byte_length(value_count, bits_per_value):
    return (value_count * bits_per_value + 7) // 8


class BitWriter:
    def __init__(self):
        self._bytes = bytearray()
        self._current = 0
        self._bits_filled = 0

    def write_bit(self, bit):
        self._current = (self._current << 1) | int(bit)
        self._bits_filled += 1
        if self._bits_filled == 8:
            self._bytes.append(self._current)
            self._current = 0
            self._bits_filled = 0

    def write_bits(self, value, bit_count):
        for shift in range(bit_count - 1, -1, -1):
            self.write_bit((value >> shift) & 1)

    def to_bytes(self):
        if self._bits_filled:
            self._bytes.append(self._current << (8 - self._bits_filled))
            self._current = 0
            self._bits_filled = 0
        return bytes(self._bytes)


class BitReader:
    def __init__(self, data):
        self._data = data
        self._byte_index = 0
        self._bit_index = 0

    def read_bit(self):
        if self._byte_index >= len(self._data):
            return 0
        byte = self._data[self._byte_index]
        bit = (byte >> (7 - self._bit_index)) & 1
        self._bit_index += 1
        if self._bit_index == 8:
            self._bit_index = 0
            self._byte_index += 1
        return bit

    def read_bits(self, bit_count):
        value = 0
        for _ in range(bit_count):
            value = (value << 1) | self.read_bit()
        return value


def pack_values(values, bits_per_value):
    writer = BitWriter()
    for value in values:
        writer.write_bits(int(value), bits_per_value)
    return writer.to_bytes()


def unpack_values(data, value_count, bits_per_value):
    reader = BitReader(data)
    values = np.zeros(value_count, dtype=np.uint32)
    for idx in range(value_count):
        values[idx] = reader.read_bits(bits_per_value)
    return value
