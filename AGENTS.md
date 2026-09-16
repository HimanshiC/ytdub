# AGENTS.md — YouTube English Dubbing Pipeline

## Project

Python CLI that takes a YouTube URL (source language: German, French, Hindi, or any
other — must not be hardcoded to one language) and produces an English-dubbed video.
Built in 2 days for an AI-agent-focused internship assignment.

## Evaluation criteria (actual rubric from the company — overrides generic framing)

Core pipeline behaviors judged: (1) Fetch & Transcribe, (2) Translate for meaning/
natural phrasing, (3) Synthesize natural, non-robotic speech that reasonably
matches the original speaker, (4) Remix & Output — replace audio via ffmpeg
without re-encoding video. Multi-speaker/voice cloning is an explicit STRETCH
goal only — attempt after the core is done, not before.

Required submission (emailed to [careers@idealabsdigital.com](mailto:careers@idealabsdigital.com), not Internshala):

* 30-minute source video + dubbed output + total processing time
* 2-hour source video + dubbed output + total processing time
* ~2-minute walkthrough video

Code is NOT reviewed at submission — only checked during a 15-min interview if
shortlisted. Working + genuinely understood beats polished, right now. Don't
sacrifice clarity for cleverness, and don't gold-plate architecture at the expense
of a working video, but also don't submit code you can't explain.

## Architecture — already decided, do not redesign

YouTube URL
→ yt-dlp (video + audio kept as separate streams/files)
→ caption check: prefer manually uploaded (`subtitles`) tracks when available;
if the selected track is missing, empty, malformed, or lacks usable timestamps,
fall back to faster-whisper. This is deterministic metadata/structure validation,
not a caption-quality classifier.
→ normalized timestamped utterance segments
→ prosody feature extraction per segment: speaking rate, pause pattern, RMS energy
envelope, coarse pitch register (see Prosody & speaker matching below)
→ group utterances into contextual translation chunks (~30–90s target) at natural
sentence/utterance boundaries, never intentionally split mid-sentence
→ Stage A: contextual LLM translation (source → natural English)
context = video title/metadata + last 1–2 translated chunks + persisted glossary
of proper nouns/terms (NOT the full transcript history)
→ duration prediction gate
→ conditionally Stage B (dub-adapt rewrite) only when predicted duration indicates
that the translation is unlikely to fit its target speech window
→ cache every stage to disk (see Caching)
→ edge-tts synthesis using a single deterministic English voice selection for each
speaker/register, measure actual duration
→ apply Sync Algorithm
→ apply bounded, smoothed energy-envelope shaping after synchronization
→ assemble audio track anchored to ORIGINAL absolute timestamps (never serial concat)
→ validate final audio duration against source video duration
→ ffmpeg mux: copy original video stream untouched, replace audio only
→ dubbed.mp4

## Canonical data model

All stages must communicate through a canonical timestamped segment representation.
At minimum each segment contains:

* stable segment ID
* original start time
* original end time
* original source text
* translated text when available
* TTS artifact path when available
* target duration
* actual TTS duration when available
* synchronization action/status
* optional prosody metadata

Timestamps and timing metadata are owned by deterministic code. LLMs must not
invent or modify segment timestamps.

## Sync algorithm — implement exactly this, do not redesign mid-task

For each segment, `target = end − start`.

1. Predict TTS duration from a calibrated chars-per-second rate for the selected
   voice. If `predicted / target <= 1.15`, generate directly and skip Stage B.
   Otherwise request one Stage B rewrite before the first TTS call.

2. Generate TTS once. Measure `actual_duration`. Compute
   `ratio = actual_duration / target`.

3. If `ratio` is in `[0.9, 1.1]`, keep as-is. Do not stretch segments that already
   fit closely.

4. If `ratio` is in `(1.1, 1.25]` or `[0.8, 0.9)`, apply pitch-preserving FFmpeg
   `atempo`, with the speed multiplier bounded to `[0.85, 1.15]`. Never exceed
   approximately 15% speed adjustment in the normal correction path.

5. If `ratio > 1.25` or `< 0.8`, trigger at most one Stage B rephrase and one
   regeneration. After regeneration, apply only the bounded `[0.85, 1.15]`
   timing correction for any residual mismatch. Never loop indefinitely.

6. Represent source pauses explicitly as silence between speech segments. If the
   generated speech is shorter than its target window, preserve the remaining
   time as silence rather than unnecessarily slowing speech.

7. Place every segment at its ORIGINAL absolute start timestamp in the final
   timeline. Never chain segments using the previous generated segment's actual
   duration. This prevents cumulative timing drift.

8. If overflow is unavoidable, consume the following silence gap first. If speech
   still overlaps the next speech segment, cap the bleed at approximately
   300–500 ms, mark the segment `SYNC_DEGRADED`, and log the event.

9. The final assembled audio must be explicitly checked against the source video
   duration before muxing. Small intentional end padding is acceptable, but large
   duration mismatch is an error that must be reported before final rendering.

## Prosody & speaker matching

Treat speaker identity and prosody/delivery as separate, decoupled modules.

### Speaker identity

Core behavior:

* choose one Edge-TTS English voice per speaker/register for the entire run;
* do not silently change voices between segments;
* keep voice selection deterministic;
* the core MVP may use a configured gender/register-matched voice.

Stretch behavior:

* XTTS or another reference-conditioned voice-cloning backend may be attempted
  only after BOTH required benchmark videos (30-min and 2-hour) are completed
  and submission-ready;
* voice cloning must remain a drop-in replacement for speaker/voice selection;
* it must not require redesigning timing, prosody, caching, or the main pipeline;
* only use source/reference audio when permitted and appropriate.

### Prosody / delivery

Do lightweight source analysis rather than building a full emotion-recognition
system.

Extract, where practical:

* speaking rate,
* pause locations/durations,
* RMS energy envelope,
* voiced duration,
* coarse pitch/range statistics.

Speaking rate and pause placement are primarily handled by the Sync Algorithm.
Do not create a separate complicated timing system.

Energy shaping:

* use the source segment's RMS envelope as a relative delivery cue;
* time-warp the envelope to the FINAL post-sync TTS duration;
* smooth the envelope before application;
* normalize it and apply only a bounded gain range;
* do not allow source noise or abrupt peaks to produce extreme TTS gain changes;
* perform this step AFTER synchronization has settled the final duration.

Pitch:

* do NOT attempt frame-level cross-lingual pitch-contour transplantation;
* do NOT attempt complex word-level emphasis transfer or syllable-cadence
  matching;
* use a stable global voice/register choice as the safe pitch/identity approximation.

The goal is to preserve broad delivery characteristics without introducing
obvious synthetic artifacts.

## Caching

Cache key = hash of the complete relevant inputs for that artifact, including:

* upstream input content,
* model/backend name and version,
* relevant configuration,
* prompt/template version,
* pipeline strategy/mode where the artifact differs between baseline and improved modes.

Do not rely solely on stage names or manually remembered `--force-*` flags.

At minimum cache:

* metadata
* transcript
* normalized segments
* prosody analysis
* translation chunks
* glossary state
* per-segment TTS audio
* synchronized segment artifacts where useful
* final assembled audio

Cache shared artifacts once and reuse them across baseline/improved modes when the
inputs are identical.

A changed translation prompt/model/configuration must invalidate only affected
downstream artifacts.

## Baseline vs. improved (feature-flag design)

There is ONE codebase and ONE infrastructure layer.

All improved behavior must be implemented as explicit strategies/configuration
flags on the same interfaces, never as a duplicate pipeline.

Shared and identical between modes:

* YouTube ingestion
* metadata
* caption/Whisper transcript acquisition
* transcript normalization
* canonical segment model
* basic file handling
* cache infrastructure
* FFmpeg mux implementation
* measurement/reporting infrastructure

### Baseline mode

Baseline should represent a reasonable straightforward implementation of the
assignment's suggested workflow:

* straightforward segment translation with no persistent context/glossary
* one-pass TTS
* no predictive Stage B adaptation
* no adaptive rephrase/measure loop
* no corrective timing optimization beyond placing generated audio at its source
  segment timestamp
* simple full-audio replacement
* shared deterministic infrastructure

The baseline must be functional and fair. Do NOT deliberately cripple it.

### Improved mode

Improved mode adds:

* contextual translation
* glossary/terminology state
* duration prediction
* conditional Stage B dubbing adaptation
* measured TTS duration
* bounded timing correction
* absolute timestamp anchoring
* explicit pause preservation
* bounded/smoothed energy-envelope shaping
* simple background ducking when enabled

The mux implementation itself should remain identical between modes so that the
comparison measures the intended pipeline differences rather than unrelated
rendering changes.

### A/B comparison

Run baseline-vs-improved comparison ONLY on the golden clip
(60–90 seconds, or the 5–10 minute integration clip).

NEVER run baseline mode on the required 30-minute or 2-hour submission videos.

The required long videos are run once with improved mode after the pipeline is
stable.

Compare using:

* timing error mean/median
* percentage of segments within ±0.2s and ±0.5s
* processing time
* Stage B trigger/rephrase count
* failed/degraded segment count

Do NOT invent a numerical translation-quality or naturalness score without a
defensible reference methodology. Use a few concrete before/after translation
examples and a listening comparison instead.

## Audio handling

Default to replacing the speech track while keeping the implementation simple
and robust.

Optional enhancement:

* simple background ducking/mixing using the original soundtrack at reduced
  level.

Do not make full Demucs/source separation a core dependency.

If background preservation is enabled, keep the algorithm bounded and fail safely
back to speech-only output rather than failing the entire pipeline.

## Long-video reliability

The system must be able to process long videos incrementally.

No stage should require loading the entire two-hour decoded audio into RAM.

Use file-based intermediate artifacts and streaming/subprocess-based media
operations wherever practical.

TTS generation must use bounded concurrency, not an unbounded
`asyncio.gather()` over hundreds of requests.

Every segment is independently retryable.

If a segment ultimately fails:

* preserve all completed artifacts,
* mark the segment failed,
* substitute silence or the safest available fallback,
* continue the run,
* report the failed segment IDs and reasons in the final summary.

## Agentic behavior

Use an explicit bounded state-machine/orchestration pattern rather than a
general-purpose autonomous agent.

### LLM judgment is appropriate for

* contextual translation
* glossary term extraction/update
* Stage B dubbing adaptation

### Deterministic code is responsible for

* download
* caption availability/structure checks
* ASR invocation
* segmentation math
* timing arithmetic
* duration measurement
* cache read/write/invalidation
* retries/backoff
* FFmpeg
* audio assembly
* final validation
* output reporting

The agentic value should come from observable decisions based on intermediate
results, not from giving an LLM control of every pipeline operation.

Log decisions such as:

```text
segment=42
target=4.20s
predicted=5.12s
decision=ADAPT_TRANSLATION

regenerated=1
new_duration=4.51s
timing_adjustment=0.93x
status=accepted
```

This decision trail is part of the evidence for the AI-agent engineering aspect.

## Style

* Plain, readable Python.
* Type hints on public functions.
* Small modules with explicit interfaces.
* No hidden global state for pipeline decisions.
* Every pipeline stage must be runnable/testable in isolation on a short clip.
* Keep deterministic operations deterministic.
* Log every Stage B trigger, regeneration decision, synchronization adjustment,
  and degraded/failure case.
* No silent failures.
* `main()` must time the full run and print total processing time clearly at the
  end.
* Prefer implementation that can be understood and explained in a 15-minute
  interview over clever abstractions.

## Implementation discipline

Work incrementally.

Always establish a working vertical slice before adding the next quality layer.

Required development progression:

1. 60–90 second ingestion/transcription smoke test
2. complete baseline dubbed video
3. improved contextual translation
4. duration-aware synchronization
5. prosody/energy shaping
6. background ducking if stable
7. final reliability hardening
8. golden-clip A/B comparison
9. 30-minute benchmark
10. 2-hour benchmark
11. walkthrough and submission packaging

Do not start optional stretch work until the improved core has successfully produced
a playable dubbed video.

Before adding any new dependency/model, consider whether it directly improves one
of the stated grading criteria and whether it is worth the implementation and
runtime risk within the deadline.

## Current status

(Update this line each session so a new agent context knows where things stand.)
