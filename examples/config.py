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
        "name": "ZipNN",
        "method": "zipnn",
        "zipnn_method": ZIPNN_METHOD,
        "verify": False,
        "description": (
            "the official `zipnn` library applied tensor by tensor with "
            '`method=ZIPNN_METHOD` and `input_format="torch"`.'
        ),
    },
]
