"""Bounded Milestone 1 orchestration: ingest, transcribe, normalize, cache."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from .dubbing import BaselineDubbingPipeline, DubbingConfig
from .cache import VideoCache, cache_key, file_sha256
from .models import Segment, VideoMetadata
from .transcription import (
    FasterWhisperProvider,
    PreferredTranscriptProvider,
    TranscriptResult,
    WhisperSettings,
)
from .youtube import MediaArtifacts, YouTubeIngestor


LOGGER = logging.getLogger(__name__)
PIPELINE_VERSION = "milestone-1-segmentation-v2"


@dataclass(frozen=True, slots=True)
class IngestionConfig:
    """Inputs that affect Milestone 1 artifact selection and cache keys."""

    url: str
    cache_root: Path = Path(".cache")
    source_language: str | None = None
    subtitle_language: str | None = None
    whisper: WhisperSettings = WhisperSettings()
    diagnostic: bool = False


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """The successfully persisted output of the Milestone 1 vertical slice."""

    video: VideoMetadata
    media: MediaArtifacts
    transcript: TranscriptResult
    segments: tuple[Segment, ...]
    cache_directory: Path

@dataclass(frozen=True, slots=True)
class DubbingPipelineResult:
    """Complete result of Milestone 2 baseline dubbing."""

    ingestion: IngestionResult
    dubbing: object
class IngestionPipeline:
    """An explicit, finite orchestration of only the first pipeline milestone."""

    def __init__(self, ingestor: YouTubeIngestor | None = None) -> None:
        self.ingestor = ingestor or YouTubeIngestor()

    def run(self, config: IngestionConfig) -> IngestionResult:
        """Run acquisition and transcription, persisting all Milestone 1 artifacts."""

        video = self.ingestor.probe(config.url)
        cache = VideoCache(config.cache_root, video.video_id)
        metadata_key = cache_key(
            {"pipeline": PIPELINE_VERSION, "metadata": video.to_dict(), "url": config.url}
        )
        cache.write_json("metadata.json", metadata_key, video.to_dict())
        LOGGER.info("Cache: wrote metadata to %s", cache.artifact_path("metadata.json"))

        media = self.ingestor.acquire_media(video, cache)
        audio_hash = file_sha256(media.audio_path)
        transcript_key = cache_key(
            {
                "pipeline": PIPELINE_VERSION,
                "audio_sha256": audio_hash,
                "video_metadata": video.to_dict(),
                "manual_subtitle_preference": config.subtitle_language or config.source_language,
                "whisper": config.whisper.to_dict(),
                "provider_order": ["youtube_manual_subtitles", "faster_whisper"],
            }
        )
        cached_transcript = cache.read_json("transcript.json", transcript_key)
        cached_segments = cache.read_json("segments.json", transcript_key)
        if not config.diagnostic and isinstance(cached_transcript, dict) and isinstance(cached_segments, list):
            transcript = TranscriptResult.from_dict(cached_transcript)
            segments = tuple(Segment.from_dict(segment) for segment in cached_segments)
            LOGGER.info("Transcript: reusing cached %s transcript (%d segments)", transcript.provider, len(segments))
        else:
            provider = PreferredTranscriptProvider(FasterWhisperProvider(config.whisper))
            transcript = provider.fetch(
                video=video,
                audio_path=media.audio_path,
                artifact_directory=cache.path,
                preferred_language=config.subtitle_language or config.source_language,
            )
            from .subtitles import normalize_cues

            segments = tuple(normalize_cues(transcript.cues))
            if not config.diagnostic:
                cache.write_json("transcript.json", transcript_key, transcript.to_dict())
                cache.write_json(
                    "segments.json", transcript_key, [segment.to_dict() for segment in segments]
                )
            if config.diagnostic:
                LOGGER.info("Diagnostic: transcript artifacts were not cached")
            else:
                LOGGER.info("Cache: wrote transcript and %d canonical segments", len(segments))

        return IngestionResult(
            video=video,
            media=media,
            transcript=transcript,
            segments=segments,
            cache_directory=cache.path,
        )
class DubbingPipeline:
    """Run Milestone 1 ingestion followed by Milestone 2 baseline dubbing."""

    def __init__(
        self,
        ingestor: YouTubeIngestor | None = None,
        dubbing_config: DubbingConfig | None = None,
    ) -> None:
        self.ingestion_pipeline = IngestionPipeline(ingestor)
        self.dubbing_pipeline = BaselineDubbingPipeline(dubbing_config)

    def run(self, config: IngestionConfig) -> DubbingPipelineResult:
        ingestion = self.ingestion_pipeline.run(config)

        cache = VideoCache(config.cache_root, ingestion.video.video_id)

        dubbing = self.dubbing_pipeline.run(
            video=ingestion.video,
            media=ingestion.media,
            segments=ingestion.segments,
            cache=cache,
        )

        return DubbingPipelineResult(
            ingestion=ingestion,
            dubbing=dubbing,
        )