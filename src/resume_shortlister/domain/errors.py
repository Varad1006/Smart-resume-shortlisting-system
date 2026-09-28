"""Error vocabulary shared by all layers."""


class ShortlisterError(Exception):
    """Base class for expected, user-facing errors."""


class InvalidInputError(ShortlisterError):
    """The request is malformed or violates a business rule."""


class PayloadTooLargeError(InvalidInputError):
    """An upload exceeds the configured size limits."""


class NotFoundError(ShortlisterError):
    """The requested entity does not exist."""


class UnsupportedDocumentError(ShortlisterError):
    """The file type is not a supported resume format."""


class ExtractionError(ShortlisterError):
    """Text could not be extracted from a document."""


class OcrUnavailableError(ExtractionError):
    """A document needs OCR but no OCR engine is reachable."""


class SocialLookupError(ShortlisterError):
    """A public profile lookup failed (rate limit, blocked, network...)."""


class InsightError(ShortlisterError):
    """The LLM could not produce a candidate summary."""
