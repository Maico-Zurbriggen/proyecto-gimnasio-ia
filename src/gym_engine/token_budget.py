from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any

from tokenizers import Tokenizer  # type: ignore[import-untyped]

# Matches the Qwen3.5 GGUF pre-tokenizer. Vocabulary and merges come from /api/show,
# never from a different model, a downloaded tokenizer or an approximate chars/token ratio.
QWEN35_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|"
    r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
TOKEN_BUDGET_VERSION = "gguf-qwen35-bpe@1"
_cache: OrderedDict[tuple[str, str, str], Tokenizer] = OrderedDict()


def cached_tokenizer(base_url: str, model: str, digest: str) -> Tokenizer | None:
    key = (base_url, model, digest)
    value = _cache.get(key)
    if value is not None:
        _cache.move_to_end(key)
    return value


def model_tokenizer(info: dict[str, Any]) -> Tokenizer:
    if info.get("tokenizer.ggml.model") != "gpt2" or info.get("tokenizer.ggml.pre") != "qwen35":
        raise ValueError("model_tokenizer_unsupported")
    tokens = info.get("tokenizer.ggml.tokens")
    merges = info.get("tokenizer.ggml.merges")
    if (
        not isinstance(tokens, list)
        or not tokens
        or not all(isinstance(item, str) for item in tokens)
        or len(set(tokens)) != len(tokens)
        or not isinstance(merges, list)
        or not all(isinstance(item, str) and len(item.split(" ")) == 2 for item in merges)
    ):
        raise ValueError("model_tokenizer_metadata_invalid")
    # No normalization, truncation, special-token replacement or padding: tokenize
    # the literal UTF-8 content just as GGUF BPE does. Counting special spellings as
    # ordinary text is conservative for the model's special-token handling.
    configuration = json.dumps(
        {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [],
            "normalizer": None,
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {
                        "type": "Split",
                        "pattern": {"Regex": QWEN35_SPLIT},
                        "behavior": "Isolated",
                        "invert": False,
                    },
                    {
                        "type": "ByteLevel",
                        "add_prefix_space": False,
                        "trim_offsets": False,
                        "use_regex": False,
                    },
                ],
            },
            "post_processor": None,
            "decoder": None,
            "model": {
                "type": "BPE",
                "dropout": None,
                "unk_token": None,
                "continuing_subword_prefix": "",
                "end_of_word_suffix": "",
                "fuse_unk": False,
                "vocab": {token: index for index, token in enumerate(tokens)},
                "merges": [item.split(" ") for item in merges],
                "byte_fallback": False,
                "ignore_merges": False,
            },
        },
        ensure_ascii=False,
    )
    try:
        return Tokenizer.from_str(configuration)
    except Exception as error:
        raise ValueError("model_tokenizer_metadata_invalid") from error


def cache_tokenizer(base_url: str, model: str, digest: str, tokenizer: Tokenizer) -> None:
    key = (base_url, model, digest)
    _cache[key] = tokenizer
    _cache.move_to_end(key)
    while len(_cache) > 2:
        _cache.popitem(last=False)


def prompt_token_bound(
    tokenizer: Tokenizer, messages: list[dict[str, str]], schema: dict[str, Any], template: str
) -> int:
    texts = [message["content"] for message in messages] + [json.dumps(schema)]
    return (
        sum(len(tokenizer.encode(text, add_special_tokens=False).ids) for text in texts)
        + len(template.encode())
        + 2048
    )
