#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

extern "C" {
#include "huf.h"
#include "zstd_internal.h"
}

namespace {

namespace nb = nanobind;

#include "common.h"
#include "codebooks.h"
#include "gpu_layout.h"
#include "ngrams.h"
#include "mixed_encode.h"
#include "binding_helpers.h"
#include "direct_storage.h"
#include "encode_queue.h"
#include "decode_queue.h"

}  // namespace

NB_MODULE(_huff0_cpp, module) {
    module.doc() = "NeuralZip stream bridge backed by upstream Zstd Huff0.";
    module.attr("MAX_BLOCK_SIZE") = kMaxBlockSize;
    module.def(
        "build_codebook_from_counts", &build_codebook_from_counts,
        nb::arg("symbol_counts"),
        nb::arg("prior_count") = 0,
        nb::arg("max_bits") = kTableLog,
        "Build a Huff0 codebook from a complete 256-bin histogram.");
    module.def(
        "build_codebook_from_header", &build_codebook_from_header,
        "Rebuild a Huff0 encoder table from its serialized header.");
    module.def(
        "build_decodebook", &build_decodebook,
        "Build a Huff0 decode table from a serialized codebook.");
    module.def(
        "prepare_ngram_codebook", &prepare_ngram_codebook,
        nb::arg("codebook"), nb::arg("vocab"), nb::arg("ngram_n"),
        "Prepare and retain the vocabulary lookup used by a Huff0 encoder.");
    module.def(
        "export_code_table", &export_code_table,
        nb::arg("codebook"),
        nb::arg("max_bits") = 12,
        "Export exact Huff0 symbol lengths and canonical code values.");
    module.def(
        "encode_gpu_tensor", &encode_gpu_tensor,
        nb::arg("raw"),
        nb::arg("codebook"),
        nb::arg("element_size"),
        nb::arg("bit_reorder"),
        nb::arg("mantissa_bits"),
        nb::arg("exponent_bits"),
        nb::arg("stream_id"),
        nb::arg("bytes_per_thread") = 8,
        nb::arg("threads_per_block") = 512,
        "Encode one byte lane with Huffman and preserve the other lanes raw.");
    module.def(
        "compress_mixed_streams_many", &compress_mixed_streams_many,
        nb::arg("tensor_specs"),
        nb::arg("min_saving_bytes") = 0,
        nb::arg("threads") = 0,
        nb::arg("compression_threshold") = 0.95,
        "Encode many tensors with one mixed-stream Huff0 native job queue.");
    module.def(
        "compress_direct_storage", &compress_direct_storage,
        nb::arg("tensor_specs"),
        nb::arg("prefix"),
        nb::arg("min_saving_bytes") = 0,
        nb::arg("threads") = 0,
        nb::arg("compression_threshold") = 0.95,
        "Encode one selected lane per tensor directly into a persistent buffer.");
    module.def(
        "decompress_mixed_streams_many", &decompress_mixed_streams_many,
        nb::arg("tensor_specs"),
        nb::arg("threads") = 0,
        "Decode many mixed-stream tensors with one native job queue.");
}
