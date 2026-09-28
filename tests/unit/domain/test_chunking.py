import pytest

from resume_shortlister.domain.chunking import chunk_text, evidence_snippet


def test_short_text_is_a_single_chunk():
    assert chunk_text("Kotlin developer with Firebase") == ["Kotlin developer with Firebase"]


def test_empty_text_has_no_chunks():
    assert chunk_text("   \n ") == []


def test_chunks_overlap_and_cover_every_word():
    words = [f"w{i}" for i in range(250)]
    chunks = chunk_text(" ".join(words), max_words=100, overlap_words=20)
    assert [len(c.split()) for c in chunks] == [100, 100, 90]
    assert chunks[1].split()[0] == "w80"  # 20-word overlap with the previous chunk
    assert chunks[-1].split()[-1] == "w249"


def test_character_budget_limits_chunks():
    chunks = chunk_text(
        " ".join(["abcdefghij"] * 50), max_words=100, overlap_words=0, max_chars=110
    )
    assert all(len(c) <= 110 for c in chunks)
    assert len(chunks) == 5


def test_unspaced_scripts_are_split_into_pieces():
    text = "简" * 200  # Chinese has no spaces between words
    chunks = chunk_text(text, max_words=3, overlap_words=0)
    assert all(len(c.replace(" ", "")) <= 120 for c in chunks)
    assert "".join(c.replace(" ", "") for c in chunks) == text


@pytest.mark.parametrize(("max_words", "overlap"), [(0, 0), (10, 10), (10, -1)])
def test_invalid_window_is_rejected(max_words, overlap):
    with pytest.raises(ValueError, match="must be"):
        chunk_text("text", max_words=max_words, overlap_words=overlap)


def test_evidence_snippet_centres_on_matching_words():
    passage = " ".join(["filler"] * 60 + ["Kotlin", "Jetpack", "Compose"] + ["filler"] * 60)
    snippet = evidence_snippet(passage, "Jetpack Compose with Kotlin", max_words=10)
    assert "Kotlin Jetpack Compose" in snippet
    assert snippet.startswith("… ") and snippet.endswith(" …")


def test_evidence_snippet_without_overlap_uses_the_start():
    passage = " ".join(f"w{i}" for i in range(100))
    assert evidence_snippet(passage, "unrelated", max_words=5) == "w0 w1 w2 w3 w4 …"
