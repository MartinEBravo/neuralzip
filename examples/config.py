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
        "method": "zipnn",
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
        "name": "NeuralZip",
        "method": "neuralzip",
        "verify": False,
        "description": "the experimental NeuralZip codec, currently a copy of local ZipNN.",
    },
]
