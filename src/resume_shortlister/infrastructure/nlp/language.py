"""Language identification with langdetect (55 languages, pure Python)."""

from __future__ import annotations

from langdetect import DetectorFactory, detect_langs
from langdetect.detector_factory import init_factory
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0  # langdetect is randomised; make results reproducible


class LangDetectLanguageDetector:
    def __init__(self, min_probability: float = 0.5, sample_chars: int = 3000) -> None:
        self._min_probability = min_probability
        self._sample_chars = sample_chars
        init_factory()  # load language profiles once, before worker threads use them

    def detect(self, text: str) -> str | None:
        sample = text[: self._sample_chars]
        if sum(ch.isalpha() for ch in sample) < 20:
            return None
        try:
            best = detect_langs(sample)[0]
        except LangDetectException:
            return None
        return best.lang if best.prob >= self._min_probability else None
