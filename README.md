# ytdub

Milestone 1 of a YouTube-to-English dubbing pipeline. This milestone downloads
separate source video/audio streams, persists video metadata, and produces a
normalized timestamped source transcript. It intentionally does not translate,
synthesize, synchronize, or mux dubbed audio.

## Setup

Use the existing project virtual environment, install the existing runtime
requirements, and install this local package:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

`yt-dlp` may require a current installation to work with YouTube. The Whisper
fallback downloads its selected model the first time it is used.

## Run

```powershell
.\.venv\Scripts\python.exe -m ytdub "https://www.youtube.com/watch?v=VIDEO_ID"
```

Useful options:

```text
--cache-dir .cache
--subtitle-language de
--source-language de
--whisper-model small
--whisper-device auto
--whisper-compute-type int8
```

The first run creates `.cache/<video-id>/` containing `metadata.json`, separate
`media/source_video.*` and `media/source_audio.*` artifacts, `transcript.json`,
and `segments.json`. A subsequent identical run reuses cached media and
transcript artifacts when their cache keys and files still match.

For the Milestone 1 smoke test, use a public video whose total duration is about
60–90 seconds; the CLI processes the complete source video.

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest
```
