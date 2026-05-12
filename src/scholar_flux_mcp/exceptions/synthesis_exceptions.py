"""Defines exceptions representing unlikely edge cases in the process of research synthesis."""


class RecordDeduplicationException(Exception):
    """Exception raised when an unexpected error is encountered during the deduplication of search records."""

    pass


class EvidenceGroundingException(Exception):
    """Exception raised when an unexpected error is encountered during the research synthesis evidence grounding."""

    pass


class EvidenceGroundingParameterException(EvidenceGroundingException):
    """Exception raised when the `GroundingService` receives invalid parameters during evidence grounding."""

    pass


__all__ = ["RecordDeduplicationException", "EvidenceGroundingException", "EvidenceGroundingParameterException"]
