EXPERIMENTS = [
    {
        "label": "GPT-2",
        "model_id": "gpt2",
    },
    {
        "label": "GPT-2 Nano",
        "model_id": "sshleifer/tiny-gpt2",
    },
    {
        "label": "Gemma 4 E2B",
        "model_id": "google/gemma-4-E2B",
    },
    {
        "label": "Gemma 4 E2B IT",
        "model_id": "google/gemma-4-E2B-it",
    },
    {
        "label": "Gemma 4 E4B",
        "model_id": "google/gemma-4-E4B",
    },
    {
        "label": "Gemma 4 E4B IT",
        "model_id": "google/gemma-4-E4B-it",
    },
    {
        "label": "DeepSeek R1",
        "model_id": "deepseek-ai/DeepSeek-R1",
    },
    {
        "label": "Llama 3.2 1B",
        "model_id": "meta-llama/Llama-3.2-1B",
    },
    {
        "label": "Llama 3.2 1B Instruct",
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
    },
    {
        "label": "Llama 3.2 3B",
        "model_id": "meta-llama/Llama-3.2-3B",
    },
    {
        "label": "Llama 3.2 3B Instruct",
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
    },
    {
        "label": "Llama 3.2 11B Vision",
        "model_id": "meta-llama/Llama-3.2-11B-Vision",
    },
    {
        "label": "Llama 3.2 11B Vision Instruct",
        "model_id": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    },
    {
        "label": "Phi-4-mini",
        "model_id": "microsoft/Phi-4-mini-instruct",
    },
    {
        "label": "Qwen 3.5 0.8B",
        "model_id": "Qwen/Qwen3.5-0.8B",
    },
    {
        "label": "Qwen 3.5 2B",
        "model_id": "Qwen/Qwen3.5-2B",
    },
    {
        "label": "Qwen 3.5 4B",
        "model_id": "Qwen/Qwen3.5-4B",
    },
    {
        "label": "Qwen 3.5 9B",
        "model_id": "Qwen/Qwen3.5-9B",
    },
]


def select_experiments(model_selectors, experiments=EXPERIMENTS):
    if not model_selectors:
        return experiments

    lookup = {}
    for experiment in experiments:
        lookup[experiment["model_id"].lower()] = experiment
        lookup[experiment["label"].lower()] = experiment

    selected = []
    missing = []
    for selector in model_selectors:
        experiment = lookup.get(selector.lower())
        if experiment is None:
            missing.append(selector)
            continue
        if experiment not in selected:
            selected.append(experiment)

    if missing:
        raise ValueError(f"Unknown model selectors: {', '.join(missing)}")

    return selected
