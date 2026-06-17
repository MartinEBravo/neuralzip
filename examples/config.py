ZIPNN_METHOD = "HUFFMAN"

MODEL_SUITE = [
    {
        "name": "GPT-2",
        "model_id": "gpt2",
        "parameters_b": 117.0,
        "default_dtype": "bfloat16",
        "load_kwargs": {"local_files_only": True},
    },
]

COMPRESSOR_SUITE = [
    {
        "name": "ZipNN local",
        "method": "zipnn-local",
        "verify": False,
        "description": (
            "the local ZipNN-compatible codec applied tensor by tensor with "
            "dtype-aware byte reordering and Huffman compression."
        ),
    },
    {
        "name": "ZipNN src",
        "method": "zipnn-src",
        "zipnn_method": ZIPNN_METHOD,
        "verify": False,
        "description": "the official ZipNN package from the authors.",
    },
    {
        "name": "NeuralZip-Fast",
        "method": "neuralzip-fast",
        "verify": False,
        "description": (
            "the fast NeuralZip codec with Jensen-Shannon "
            "clustered codebooks reused by byte stream."
        ),
    },
    {
        "name": "NeuralZip-Ratio",
        "method": "neuralzip-ratio",
        "verify": False,
        "description": (
            "the codebook-aware exponent-stream ratio experiment selecting "
            "between ZipNN-local Huffman and full-vocabulary n-gram arithmetic."
        ),
    },
    {
        "name": "Global Huffman",
        "method": "global-huffman",
        "verify": False,
        "description": (
            "an exponent-stream baseline that builds one Huffman tree from a "
            "representative chunk and reuses it for the remaining chunks."
        ),
    },
]
