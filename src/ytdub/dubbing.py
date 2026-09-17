"""Baseline translation, TTS, timestamped audio assembly, and video muxing."""

from __future__ import annotations
import subprocess
import asyncio
from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import subprocess
from typing import Iterable

import edge_tts
# from deep_translator import GoogleTranslator
from .translator import HybridTranslator
from .cache import VideoCache, cache_key
from .errors import RuntimeDependencyError
from .models import Segment, VideoMetadata
from .youtube import MediaArtifacts


LOGGER = logging.getLogger(__name__)

DEFAULT_VOICE = "en-US-AriaNeural"
# TRANSLATION_BACKEND = "googletrans-deep-translator-v1"
TRANSLATION_BACKEND = "groq-gpt-oss-20b-v1"
TTS_BACKEND = "edge-tts-v1"
MAX_TTS_CONCURRENCY = 4


@dataclass(frozen=True, slots=True)
class DubbingConfig:
    voice: str = DEFAULT_VOICE
    tts_concurrency: int = MAX_TTS_CONCURRENCY


@dataclass(frozen=True, slots=True)
class DubbingResult:
    segments: tuple[Segment, ...]
    audio_path: Path
    video_path: Path
    output_path: Path
    duration: float


class BaselineDubbingPipeline:
    """Straightforward per-segment baseline used before adaptive synchronization."""

    def __init__(self, config: DubbingConfig | None = None) -> None:
        self.config = config or DubbingConfig()

    def run(
        self,
        *,
        video: VideoMetadata,
        media: MediaArtifacts,
        segments: Iterable[Segment],
        cache: VideoCache,
    ) -> DubbingResult:
        segment_list = list(segments)
        if not segment_list:
            raise ValueError("cannot dub an empty segment list")

        segment_list = self._merge_short_segments(segment_list)

        context_segments = self._build_context_segments(segment_list)

        LOGGER.info(
            "Built %d contextual translation groups from %d transcript segments",
            len(context_segments),
            len(segment_list),
        )

        translated = self._translate_segments(
            video,
            context_segments,
            cache,
        )

        spoken = self._synthesize_segments(
            translated,
            cache,
        )
        audio_path = self._assemble_audio(
            video=video,
            segments=spoken,
            cache=cache,
        )

        output_path = cache.artifact_path("output/dubbed_baseline.mp4")
        self._mux_video(
            video_path=media.video_path,
            audio_path=audio_path,
            output_path=output_path,
        )

        duration = _probe_duration(output_path)

        # Persist the complete baseline segment state for later milestones.
        key = cache_key(
            {
                "pipeline": "milestone-2-baseline",
                "video_id": video.video_id,
                "voice": self.config.voice,
                "translation_backend": TRANSLATION_BACKEND,
                "tts_backend": TTS_BACKEND,
                "segments": [segment.to_dict() for segment in spoken],
            }
        )
        cache.write_json(
            "dub_segments.json",
            key,
            [segment.to_dict() for segment in spoken],
        )

        return DubbingResult(
            segments=tuple(spoken),
            audio_path=audio_path,
            video_path=media.video_path,
            output_path=output_path,
            duration=duration,
        )
        
    def _merge_short_segments(
        self,
        segments: list[Segment],
        *,
        min_duration: float = 0.6,
    ) -> list[Segment]:
        if not segments:
            return segments

        merged: list[Segment] = []

        for segment in segments:
            duration = segment.end - segment.start

            if merged and duration < min_duration:
                previous = merged[-1]

                previous.end = segment.end
                previous.source_text = (
                    f"{previous.source_text} {segment.source_text}"
                ).strip()

                previous.target_duration = (
                    previous.end - previous.start
                )

                LOGGER.info(
                    "SEGMENT MERGE %s -> %s duration=%.3fs",
                    segment.segment_id,
                    previous.segment_id,
                    duration,
                )
            else:
                merged.append(segment)

        return merged
    def _build_context_segments(
        self,
        segments: list[Segment],
        *,
        min_duration: float = 15.0,
        max_duration: float = 30.0,
        max_chars: int = 1600,
    ) -> list[Segment]:
        """Group short transcript segments into coherent translation/TTS units."""

        if not segments:
            return []

        groups: list[Segment] = []
        current: list[Segment] = []

        def flush() -> None:
            if not current:
                return

            first = current[0]
            last = current[-1]

            source_text = " ".join(
                segment.source_text.strip()
                for segment in current
                if segment.source_text.strip()
            ).strip()

            if not source_text:
                current.clear()
                return

            group_id = (
                f"context-{first.segment_id}-{last.segment_id}"
            )

            group_duration = last.end - first.start
            
            groups.append(
                Segment(
                    segment_id=group_id,
                    start=first.start,
                    end=last.end,
                    source_text=source_text,
                    target_duration=group_duration,
                )
            )

            LOGGER.info(
                "CONTEXT GROUP %s duration=%.3fs segments=%d chars=%d",
                group_id,
                group_duration,
                len(current),
                len(source_text),
            )

            current.clear()

        for segment in segments:
            if not segment.source_text.strip():
                continue

            if not current:
                current.append(segment)
                continue

            first = current[0]
            proposed_end = segment.end
            proposed_duration = proposed_end - first.start

            current_text = " ".join(
                item.source_text.strip()
                for item in current
                if item.source_text.strip()
            )

            proposed_text = (
                f"{current_text} {segment.source_text.strip()}"
            ).strip()

            previous_ends_sentence = (
                current[-1].source_text.strip().endswith(
                    (".", "!", "?", "。", "！", "？")
                )
            )

            if (
                proposed_duration > min_duration
                and previous_ends_sentence
            ):
                flush()
                current.append(segment)
                continue

            if (
                proposed_duration > max_duration
                or len(proposed_text) > max_chars
            ):
                flush()
                current.append(segment)
                continue

            current.append(segment)

        flush()
        return groups
    def _translate_segments(
        self,
        video: VideoMetadata,
        segments: list[Segment],
        cache: VideoCache,
    ) -> list[Segment]:
        LOGGER.info("Translation: translating %d segments", len(segments))

        source_language = (
            video.language
            or video.original_language
            or "auto"
        ).split("-")[0].lower()

        translator = HybridTranslator(
            source_language=source_language,
        )

        result = list(segments)
        pending: list[tuple[int, Segment, str]] = []

        # First reuse anything already cached.
        for index, segment in enumerate(segments):
            key = cache_key(
                {
                    "backend": translator.provider_name,
                    "source_language": source_language,
                    "target_language": "en",
                    "mode": "baseline",
                    "source_text": segment.source_text,
                }
            )

            cached = cache.read_json(
                f"translation/{segment.segment_id}.json",
                key,
            )

            if (
                isinstance(cached, dict)
                and isinstance(cached.get("translated_text"), str)
                and cached["translated_text"].strip()
            ):
                translated_text = cached["translated_text"].strip()
                result[index].translated_text = translated_text

                LOGGER.info(
                    "Translation [%d/%d] reused %s",
                    index + 1,
                    len(segments),
                    segment.segment_id,
                )
            else:
                pending.append((index, segment, key))

        # Translate uncached segments in one batch.
        if pending:
            pending_segments = [segment for _, segment, _ in pending]

            LOGGER.info(
                "Translation: translating %d uncached segments with %s",
                len(pending_segments),
                translator.provider_name,
            )

            translated_batch = translator.translate_segments(
                pending_segments,
            )

            if len(translated_batch) != len(pending):
                raise RuntimeError(
                    "translation backend returned an unexpected number of results: "
                    f"expected {len(pending)}, got {len(translated_batch)}"
                )

            for (index, segment, key), translated_text in zip(
                pending,
                translated_batch,
                strict=True,
            ):
                translated_text = translated_text.strip()

                if not translated_text:
                    raise RuntimeError(
                        f"translation returned empty text for {segment.segment_id}"
                    )

                result[index].translated_text = translated_text

                cache.write_json(
                    f"translation/{segment.segment_id}.json",
                    key,
                    {
                        "segment_id": segment.segment_id,
                        "source_language": source_language,
                        "source_text": segment.source_text,
                        "translated_text": translated_text,
                        "mode": "baseline",
                        "backend": translator.provider_name,
                    },
                )

        # Print the final mapping in canonical segment order.
        for index, segment in enumerate(result):
            LOGGER.info(
                "Translation [%d/%d] %s -> %s",
                index + 1,
                len(result),
                segment.source_text,
                segment.translated_text,
            )

        return result
    #older implementation of translation using GoogleTranslator, now commented out
    # def _translate_segments(
    #     self,
    #     video: VideoMetadata,
    #     segments: list[Segment],
    #     cache: VideoCache,
    # ) -> list[Segment]:
    #     LOGGER.info("Translation: translating %d segments", len(segments))

    #     source_language = (
    #         video.language
    #         or video.original_language
    #         or "auto"
    #     ).split("-")[0].lower()

    #     pending: list[tuple[int, Segment]] = []
    #     result = list(segments)

    #     # First reuse anything already cached.
    #     for index, segment in enumerate(segments):
    #         key = cache_key(
    #             {
    #                 "backend": TRANSLATION_BACKEND,
    #                 "source_language": source_language,
    #                 "target_language": "en",
    #                 "source_text": segment.source_text,
    #             }
    #         )

    #         cached = cache.read_json(
    #             f"translation/{segment.segment_id}.json",
    #             key,
    #         )

    #         if (
    #             isinstance(cached, dict)
    #             and isinstance(cached.get("translated_text"), str)
    #             and cached["translated_text"].strip()
    #         ):
    #             result[index].translated_text = cached["translated_text"].strip()
    #         else:
    #             pending.append((index, segment))

    #     if not pending:
    #         LOGGER.info("Translation: all segments reused from cache")
    #         return result

    #     translator = GoogleTranslator(
    #         source=source_language,
    #         target="en",
    #     )

    #     texts = [segment.source_text for _, segment in pending]

    #     LOGGER.info(
    #         "Translation: batch translating %d uncached segments",
    #         len(texts),
    #     )

    #     translated_batch: list[str | None] = []

    #     for attempt in range(3):
    #         try:
    #             translated_batch = translator.translate_batch(texts)
    #             break
    #         except Exception as error:
    #             if attempt == 2:
    #                 raise RuntimeError(
    #                     f"batch translation failed after 3 attempts: {error}"
    #                 ) from error

    #             wait_seconds = 2 ** attempt
    #             LOGGER.warning(
    #                 "Translation: batch request failed, retrying in %ds: %s",
    #                 wait_seconds,
    #                 error,
    #             )
    #             import time
    #             time.sleep(wait_seconds)

    #     if len(translated_batch) != len(pending):
    #         raise RuntimeError(
    #             "translation backend returned an unexpected number of results: "
    #             f"expected {len(pending)}, got {len(translated_batch)}"
    #         )

    #     for (index, segment), translated_text in zip(
    #         pending,
    #         translated_batch,
    #         strict=True,
    #     ):
    #         if not translated_text or not translated_text.strip():
    #             raise RuntimeError(
    #                 f"translation returned empty text for {segment.segment_id}"
    #             )

    #         translated_text = translated_text.strip()
    #         result[index].translated_text = translated_text

    #         key = cache_key(
    #             {
    #                 "backend": TRANSLATION_BACKEND,
    #                 "source_language": source_language,
    #                 "target_language": "en",
    #                 "source_text": segment.source_text,
    #             }
    #         )

    #         cache.write_json(
    #             f"translation/{segment.segment_id}.json",
    #             key,
    #             {
    #                 "segment_id": segment.segment_id,
    #                 "source_text": segment.source_text,
    #                 "translated_text": translated_text,
    #             },
    #         )

    #     for index, segment in enumerate(result, start=1):
    #         LOGGER.info(
    #             "Translation [%d/%d] %s -> %s",
    #             index,
    #             len(result),
    #             segment.source_text,
    #             segment.translated_text,
    #         )

    #     return result

    def _synthesize_segments(
        self,
        segments: list[Segment],
        cache: VideoCache,
    ) -> list[Segment]:
        LOGGER.info(
            "TTS: synthesizing %d segments with voice=%s concurrency=%d",
            len(segments),
            self.config.voice,
            self.config.tts_concurrency,
        )

        async def run_all() -> None:
            semaphore = asyncio.Semaphore(max(1, self.config.tts_concurrency))

            async def one(segment: Segment, index: int) -> None:
                if not segment.translated_text:
                    raise ValueError(f"segment {segment.segment_id} has no translated text")

                key = cache_key(
                    {
                        "backend": TTS_BACKEND,
                        "voice": self.config.voice,
                        "text": segment.translated_text,
                    }
                )

                cached = cache.read_json(
                    f"tts/{segment.segment_id}.json",
                    key,
                )

                if isinstance(cached, dict):
                    path_value = cached.get("artifact_path")
                    duration_value = cached.get("duration")
                    if (
                        isinstance(path_value, str)
                        and isinstance(duration_value, (int, float))
                        and Path(path_value).is_file()
                    ):
                        segment.tts_artifact_path = path_value
                        segment.actual_tts_duration = float(duration_value)
                        LOGGER.info(
                            "TTS [%d/%d] reused %s",
                            index,
                            len(segments),
                            segment.segment_id,
                        )
                        return

                async with semaphore:
                    output_path = cache.artifact_path(
                        f"tts/{segment.segment_id}.mp3"
                    )
                    communicate = edge_tts.Communicate(
                        segment.translated_text,
                        self.config.voice,
                    )
                    await communicate.save(str(output_path))

                duration = _probe_duration(output_path)

                segment.tts_artifact_path = str(output_path)
                segment.actual_tts_duration = duration

                cache.write_json(
                    f"tts/{segment.segment_id}.json",
                    key,
                    {
                        "segment_id": segment.segment_id,
                        "artifact_path": str(output_path),
                        "duration": duration,
                    },
                )

                LOGGER.info(
                    "TTS [%d/%d] %s duration=%.3fs target=%.3fs",
                    index,
                    len(segments),
                    segment.segment_id,
                    duration,
                    segment.target_duration or 0.0,
                )

            await asyncio.gather(
                *(one(segment, index) for index, segment in enumerate(segments, start=1))
            )

        asyncio.run(run_all())

        for segment in segments:
            self._synchronize_segment(
                segment,
                cache,
            )
        return segments
    def _synchronize_segment(
        self,
        segment: Segment,
        cache: VideoCache,
    ) -> None:
        if not segment.tts_artifact_path:
            raise RuntimeError(
                f"segment {segment.segment_id} has no TTS artifact"
            )

        target = segment.target_duration or 0.0
        actual = segment.actual_tts_duration or 0.0

        if target <= 0.0:
            segment.synchronization_status = "SKIPPED"
            segment.synchronization_action = "ZERO_TARGET"
            return

        if actual <= 0.0:
            raise RuntimeError(
                f"segment {segment.segment_id} has invalid TTS duration"
            )

        ratio = actual / target

        # Close enough. Keep the original TTS.
        if 0.90 <= ratio <= 1.10:
            segment.synchronization_status = "SYNCED"
            segment.synchronization_action = "KEEP"

            LOGGER.info(
                "SYNC %s action=KEEP actual=%.3fs target=%.3fs ratio=%.3f",
                segment.segment_id,
                actual,
                target,
                ratio,
            )
            return

        # Mild/moderate mismatch: bounded time stretching.
        if 0.72 <= ratio <= 1.25:
            atempo = actual / target
            atempo = max(0.72, min(1.15, atempo))
            # atempo = min(0.72, max(1.15, ratio))

            raw_path = Path(segment.tts_artifact_path)
            corrected_path = cache.artifact_path(
                f"sync/{segment.segment_id}.mp3"
            )

            key = cache_key(
                {
                    "stage": "sync",
                    "segment_id": segment.segment_id,
                    "source_tts": str(raw_path),
                    "target_duration": round(target, 3),
                    "actual_duration": round(actual, 3),
                    "atempo": round(atempo, 4),
                }
            )

            cached = cache.read_json(
                f"sync/{segment.segment_id}.json",
                key,
            )

            if (
                isinstance(cached, dict)
                and isinstance(cached.get("artifact_path"), str)
                and Path(cached["artifact_path"]).is_file()
                and isinstance(cached.get("duration"), (int, float))
            ):
                segment.tts_artifact_path = cached["artifact_path"]
                segment.actual_tts_duration = float(cached["duration"])
                segment.synchronization_status = "SYNCED"
                segment.synchronization_action = (
                    f"ATEMPO:{atempo:.3f}"
                )

                LOGGER.info(
                    "SYNC %s reused atempo=%.3f duration=%.3fs target=%.3fs",
                    segment.segment_id,
                    atempo,
                    segment.actual_tts_duration,
                    target,
                )
                return

            command = [
                "ffmpeg",
                "-y",
                "-i",
                str(raw_path),
                "-filter:a",
                f"atempo={atempo:.6f}",
                "-c:a",
                "libmp3lame",
                "-q:a",
                "2",
                str(corrected_path),
            ]

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"sync correction failed for {segment.segment_id}: "
                    f"{result.stderr[-1000:]}"
                )

            corrected_duration = _probe_duration(corrected_path)

            segment.tts_artifact_path = str(corrected_path)
            segment.actual_tts_duration = corrected_duration
            segment.synchronization_status = "SYNCED"
            segment.synchronization_action = (
                f"ATEMPO:{atempo:.3f}"
            )

            cache.write_json(
                f"sync/{segment.segment_id}.json",
                key,
                {
                    "segment_id": segment.segment_id,
                    "artifact_path": str(corrected_path),
                    "duration": corrected_duration,
                    "target_duration": target,
                    "atempo": atempo,
                },
            )

            LOGGER.info(
                "SYNC %s action=ATEMPO:%.3f actual=%.3fs target=%.3fs",
                segment.segment_id,
                atempo,
                corrected_duration,
                target,
            )
            return

        # Severe mismatch. Do not apply ridiculous atempo values.
        # Stage B rephrasing will handle these later.
        segment.synchronization_status = "DEGRADED"
        segment.synchronization_action = (
            f"STAGE_B_REQUIRED:RATIO:{ratio:.3f}"
        )

        LOGGER.warning(
            "SYNC degraded %s actual=%.3fs target=%.3fs ratio=%.3f",
            segment.segment_id,
            actual,
            target,
            ratio,
        )

    def _assemble_audio(
        self,
        *,
        video: VideoMetadata,
        segments: list[Segment],
        cache: VideoCache,
    ) -> Path:
        output_path = cache.artifact_path("output/dubbed_audio.m4a")

        duration = (
            float(video.duration)
            if video.duration is not None
            else max(
                segment.end
                for segment in segments
            )
        )

        ffmpeg_inputs: list[str] = [
            "-f",
            "lavfi",
            "-t",
            f"{duration:.3f}",
            "-i",
            "anullsrc=r=48000:cl=stereo",
        ]

        filters: list[str] = ["[0:a]aformat=sample_rates=48000:channel_layouts=stereo[base]"]

        mix_labels = ["[base]"]

        for index, segment in enumerate(segments, start=1):
            if not segment.tts_artifact_path:
                continue

            ffmpeg_inputs.extend(["-i", segment.tts_artifact_path])

            delay_ms = max(0, round(segment.start * 1000))

            filters.append(
                f"[{index}:a]"
                f"aformat=sample_rates=48000:channel_layouts=stereo,"
                f"adelay={delay_ms}|{delay_ms}"
                f"[seg{index}]"
            )
            mix_labels.append(f"[seg{index}]")

        if len(mix_labels) == 1:
            filter_complex = "[base]anull"
        else:
            filter_complex = (
                "".join(mix_labels)
                + f"amix=inputs={len(mix_labels)}:duration=first:dropout_transition=0,"
                f"atrim=duration={duration:.3f},asetpts=N/SR/TB[out]"
            )

        command = [
            "ffmpeg",
            "-y",
            *ffmpeg_inputs,
            "-filter_complex",
            ";".join(filters + [filter_complex]),
            "-map",
            "[out]" if len(mix_labels) > 1 else "[base]",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-t",
            f"{duration:.3f}",
            str(output_path),
        ]

        LOGGER.info("Audio: assembling timestamped baseline dub")
        _run_ffmpeg(command)

        LOGGER.info(
            "Audio: baseline dub ready duration=%.3fs path=%s",
            _probe_duration(output_path),
            output_path,
        )
        return output_path

    def _mux_video(
        self,
        *,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
    ) -> None:
        LOGGER.info("Mux: replacing original audio and copying video stream")

        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ]

        _run_ffmpeg(command)

        LOGGER.info("Mux: output ready at %s", output_path)


def _translate_text(text: str, *, source_language: str | None) -> str:
    source = (source_language or "auto").split("-")[0].lower()

    try:
        translator = GoogleTranslator(
            source=source,
            target="en",
        )
        translated = translator.translate(text)
    except Exception as error:
        raise RuntimeError(
            f"translation failed for {text!r}: {error}"
        ) from error

    if not translated or not translated.strip():
        raise RuntimeError(f"translation returned empty text for {text!r}")

    return translated.strip()


def _probe_duration(path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeDependencyError(
            "ffprobe is required for duration measurement."
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"ffprobe failed for {path}: {error.stderr.strip()}"
        ) from error

    return float(result.stdout.strip())


def _run_ffmpeg(command: list[str]) -> None:
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeDependencyError(
            "ffmpeg is required for audio assembly and video muxing."
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "FFmpeg failed:\n"
            f"{error.stderr[-4000:]}"
        ) from error