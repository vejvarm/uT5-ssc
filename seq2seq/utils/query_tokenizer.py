from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from tokenizers import AddedToken
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

T5_QUERY_EXTRA_TOKENS = ("{", "}", " <=", " <", "^^")


def add_t5_query_tokens(tokenizer: PreTrainedTokenizerFast) -> None:
    if isinstance(tokenizer, T5TokenizerFast):
        tokenizer.add_tokens(
            [AddedToken(token, normalized=True) for token in T5_QUERY_EXTRA_TOKENS]
        )


def clean_decoded_query(text: str) -> str:
    return (
        text.replace("</s>", "")
        .replace("<pad>", "")
        .replace("{{", "{")
        .replace("}}", "}")
        .replace("< >", "<>")
        .strip()
    )


def decode_query_tokens(
    tokenizer: PreTrainedTokenizerFast,
    token_ids: Iterable[int],
    *,
    skip_special_tokens: bool,
    **decode_kwargs: Any,
) -> str:
    decoded = tokenizer.decode(
        token_ids,
        skip_special_tokens=skip_special_tokens,
        **decode_kwargs,
    )
    return clean_decoded_query(decoded)


def batch_decode_query_tokens(
    tokenizer: PreTrainedTokenizerFast,
    token_ids: Iterable[Iterable[int]],
    *,
    skip_special_tokens: bool,
    **decode_kwargs: Any,
) -> list[str]:
    decoded = tokenizer.batch_decode(
        token_ids,
        skip_special_tokens=skip_special_tokens,
        **decode_kwargs,
    )
    return [clean_decoded_query(text) for text in decoded]
