"""Split resume text into overlapping passages.

Transformer encoders truncate long inputs (512 tokens here), so embedding a whole resume
silently ignores everything after the first page or so. Scoring overlapping passages and
keeping the best one lets skills listed anywhere in the resume count.
"""

from __future__ import annotations

# Scripts written without spaces (Chinese, Japanese, Thai...) produce very long
# whitespace-delimited "words"; break them so passages stay within the model window.
_MAX_TOKEN_CHARS = 40


def _tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for word in text.split():
        if len(word) <= _MAX_TOKEN_CHARS:
            tokens.append(word)
        else:
            tokens.extend(
                word[i : i + _MAX_TOKEN_CHARS] for i in range(0, len(word), _MAX_TOKEN_CHARS)
            )
    return tokens


def chunk_text(
    text: str, max_words: int = 50, overlap_words: int = 15, max_chars: int | None = None
) -> list[str]:
    """Return overlapping passages of at most ``max_words`` words / ``max_chars`` characters.

    ``max_chars`` defaults to 12 characters per word, which only bites for scripts without
    spaces, keeping such passages inside the encoder's token window.
    """
    if max_words < 1:
        raise ValueError("max_words must be >= 1")
    if not 0 <= overlap_words < max_words:
        raise ValueError("overlap_words must be >= 0 and smaller than max_words")

    if max_chars is None:
        max_chars = max_words * 12
    tokens = _tokens(text)
    if not tokens:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = start
        length = 0
        while end < len(tokens) and end - start < max_words:
            extra = len(tokens[end]) + (1 if end > start else 0)
            if end > start and length + extra > max_chars:
                break
            length += extra
            end += 1
        chunks.append(" ".join(tokens[start:end]))
        if end >= len(tokens):
            break
        start = max(end - overlap_words, start + 1)
    return chunks


_WORD_CHARS = ".,;:!?()[]{}\"'`|/\\<>*_#"


def _normalise_word(word: str) -> str:
    return word.strip(_WORD_CHARS).lower()


def evidence_snippet(passage: str, query: str, max_words: int = 45) -> str:
    """Return the window of ``passage`` that shares the most words with ``query``.

    The reranker scores a whole passage; this picks the part worth showing to a human.
    Falls back to the start of the passage when nothing overlaps (e.g. cross-lingual).
    """
    words = passage.split()
    if len(words) <= max_words:
        return passage.strip()
    keywords = {w for w in map(_normalise_word, query.split()) if len(w) >= 3}
    hits = [1 if _normalise_word(w) in keywords else 0 for w in words]

    best_start, best_hits = 0, sum(hits[:max_words])
    window = best_hits
    for start in range(1, len(words) - max_words + 1):
        window += hits[start + max_words - 1] - hits[start - 1]
        if window > best_hits:
            best_start, best_hits = start, window

    snippet = " ".join(words[best_start : best_start + max_words])
    prefix = "… " if best_start > 0 else ""
    suffix = " …" if best_start + max_words < len(words) else ""
    return f"{prefix}{snippet}{suffix}"
