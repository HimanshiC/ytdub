"""Explicit errors raised by the ingestion/transcription layer."""


class IngestionError(RuntimeError):
    """A recoverable, user-facing ingestion pipeline failure."""


class RuntimeDependencyError(IngestionError):
    """A required optional runtime package or external executable is absent."""


class TranscriptUnavailable(IngestionError):
    """A transcript provider could not produce a usable transcript."""


class SubtitleParseError(TranscriptUnavailable):
    """A subtitle file was empty, malformed, or did not contain timed text."""
