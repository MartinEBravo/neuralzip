

MODEL_SUITE = [
    {
        "name": "Latam-GPT 70B",
        "model_id": "latam-gpt/Llama-3.1-70B-LatamGPT-SFT-1.0",
        "parameters_b": 70.0,
        "default_dtype": "bfloat16",
    }
#     {
#         "name": "Llama 3.3 70B Instruct",
#         "model_id": "meta-llama/Llama-3.3-70B-Instruct",
#         "parameters_b": 70.0,
#         "default_dtype": "bfloat16",
#     },
#     {
#         "name": "Gemma 4 31B",
#         "model_id": "google/gemma-4-31B",
#         "parameters_b": 31.0,
#         "default_dtype": "bfloat16",
#     },
#     {
#         "name": "Phi-4 14B",
#         "model_id": "microsoft/phi-4",
#         "parameters_b": 14.0,
#         "default_dtype": "bfloat16",
#     },
#     {
#         "name": "Qwen3.5 27B",
#         "model_id": "Qwen/Qwen3.5-27B",
#         "parameters_b": 27.0,
#         "default_dtype": "bfloat16",
#     },
]

COMPRESSOR_SUITE = [
    {
        "name": "ZipNN",
        "description": 'the official `zipnn` library applied tensor by tensor with `method=ZIPNN_METHOD` and `input_format="torch"`.',
    },
    {
        "name": "NeuralZip",
        "description": "clustered tensor-level exponent n-gram Huffman coding, with signs and mantissas stored verbatim.",
    },
]