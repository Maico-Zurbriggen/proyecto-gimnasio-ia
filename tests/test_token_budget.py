from typing import Any

from pytest import raises
from tokenizers.pre_tokenizers import ByteLevel

from gym_engine.token_budget import (
    cache_tokenizer,
    cached_tokenizer,
    model_tokenizer,
    prompt_token_bound,
)


def tokenizer_metadata() -> dict[str, Any]:
    tokens = sorted(ByteLevel.alphabet())
    merges = []
    for power in range(1, 10):
        half = "a" * (2 ** (power - 1))
        tokens.append(half * 2)
        merges.append(f"{half} {half}")
    return {
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen35",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.merges": merges,
    }


def test_gguf_bpe_uses_exact_merges_without_normalizing_or_truncating() -> None:
    tokenizer = model_tokenizer(tokenizer_metadata())
    assert len(tokenizer.encode("a" * 512).ids) == 1
    assert len(tokenizer.encode("a" * 1024).ids) == 2
    assert len(tokenizer.encode("é").ids) == 2
    assert len(tokenizer.encode("e\u0301").ids) == 3
    assert prompt_token_bound(tokenizer, [{"content": "a" * 512}], {}, "x") == 2052


def test_tokenizer_cache_is_bounded_and_separates_model_revisions_and_endpoints() -> None:
    tokenizer = model_tokenizer(tokenizer_metadata())
    cache_tokenizer("https://budget.test", "model", "first", tokenizer)
    assert cached_tokenizer("https://budget.test", "model", "first") is tokenizer
    assert cached_tokenizer("https://other.test", "model", "first") is None
    assert cached_tokenizer("https://budget.test", "another-model", "first") is None
    assert cached_tokenizer("https://budget.test", "model", "second") is None
    cache_tokenizer("https://budget.test", "model", "second", tokenizer)
    cache_tokenizer("https://budget.test", "model", "third", tokenizer)
    assert cached_tokenizer("https://budget.test", "model", "first") is None


def test_unknown_or_incomplete_tokenizer_metadata_is_rejected() -> None:
    info = tokenizer_metadata()
    with raises(ValueError, match="unsupported"):
        model_tokenizer({**info, "tokenizer.ggml.pre": "unknown"})
    with raises(ValueError, match="metadata_invalid"):
        model_tokenizer({**info, "tokenizer.ggml.tokens": ["a", "a"]})
    with raises(ValueError, match="metadata_invalid"):
        model_tokenizer({**info, "tokenizer.ggml.merges": ["absent token"]})
