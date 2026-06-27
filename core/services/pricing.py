"""Token cost estimation.

Prices are USD per 1K tokens (approximate list prices; adjust as needed). Unknown
models cost 0 so the deterministic fake provider doesn't distort accounting.
"""
from __future__ import annotations

# model -> (input_per_1k, output_per_1k)
CHAT_PRICES = {
    "gpt-4o-mini": (0.00015, 0.00060),
    "gpt-4o": (0.00250, 0.01000),
    "gpt-4.1-mini": (0.00040, 0.00160),
}

# model -> input_per_1k (embeddings are input-only)
EMBEDDING_PRICES = {
    "text-embedding-3-small": 0.00002,
    "text-embedding-3-large": 0.00013,
}


def chat_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    in_rate, out_rate = CHAT_PRICES.get(model, (0.0, 0.0))
    return (prompt_tokens / 1000) * in_rate + (completion_tokens / 1000) * out_rate


def embedding_cost(model: str, tokens: int) -> float:
    return (tokens / 1000) * EMBEDDING_PRICES.get(model, 0.0)


def estimate_cost(
    *,
    chat_model: str,
    prompt_tokens: int,
    completion_tokens: int,
    embedding_model: str = "",
    embedding_tokens: int = 0,
) -> float:
    total = chat_cost(chat_model, prompt_tokens, completion_tokens)
    if embedding_model:
        total += embedding_cost(embedding_model, embedding_tokens)
    return round(total, 8)
