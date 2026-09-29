# PSYCON Diarization Benchmark

An independent, local web application for **benchmarking and diagnosing [NVIDIA Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)** on video and audio recordings.

This is an **experiment tool, not a transcription app.** Its purpose is to make the raw behaviour of the model inspectable, so you can work out *where* a diarization failure originated, rather than only seeing a final transcript.

> The benchmark is fully independent of the existing PSYCON pipeline. It does not import, call, or modify it.

---

## Why this exists

In PSYCON recordings, Nemotron 3 Diarization appeared to produce only **13 usable speaker profiles** across recordings where **85 people were marked**. That number alone does not say *why*. The gap could originate in any layer of the stack:

```
Nemotron model
  → audio preprocessing
  → inference configuration
  → streaming state
  → overlap handling
  → speaker capacity (max 8 channels)
  → ASR
  → downstream processing
```

This app keeps each of those layers separately inspectable and produces evidence-backed diagnostics for each, so the result can be attributed to the correct layer.

### A note on terminology

"Marked people" and "speaker channels" are **different quantities** and are never treated as interchangeable.

| Say this | Not this |
|---|---|
| "13 Nemotron speaker channels were observed across the evaluated recordings" | "Nemotron detected only 13 of 85 speakers" |
| "85 people were marked in the PSYCON dataset" | "There were 85 speakers Nemotron should have found" |

Nemotron outputs at most **8 channels at a time**, and only people who actually speak audibly (and at most 8 simultaneously) can be represented. The app helps you measure the relationship between marked people, actual participants, audible speakers, and output channels, rather than assuming they should be equal.

---

## Features

### Input and preprocessing
- Accepts MP4, MOV, WAV, MP3, M4A, and other formats FFmpeg can read.
- For video, audio is extracted with FFmpeg and the **original video is preserved** for playback alongside the timeline.
- Audio is explicitly normalized to **mono, 16 kHz, 16-bit PCM WAV**, and the resulting audio properties are logged.

### Nemotron inference (no substitutions)
- Runs `nvidia/Nemotron-3-Diarization` directly. **No fallback model is ever used.**
- If CUDA is unavailable, the app reports it clearly instead of silently changing anything.
- Two explicit modes:
  - **Offline benchmark:** the recommended offline-style configuration (primary benchmark).
  - **Streaming benchmark:** configurable chunk, context, FIFO, and speaker-cache parameters, with speaker state carried across chunks (identities are never reset per chunk).
- Every inference parameter is logged: model name/version, mode, chunk length, right context, FIFO length, speaker-cache settings, output frame resolution, speaker count, hardware, and inference time.

### Raw output preserved
- Raw diarization output is stored **before any postprocessing** and is never modified.
- Per segment: speaker ID, start, end, duration, and activity probability where available.
- Per-frame speaker activity probabilities (`[T, 8]` matrix) are kept where practical.
- Exports: JSON, CSV, RTTM.

### Anonymous speaker labels
Nemotron's labels are anonymous. They appear as `Speaker 0` … `Speaker 7`, stay consistent across the whole recording, and are **never** treated as real people or assumed to match `Person 1`, `Person 2`, etc.

### Speaker-attributed transcript
ASR is run *after* diarization. Each transcript segment shows timestamp, speaker ID, text, and confidence (if available):

```
[00:04.210 - 00:08.730]
Speaker 0:
"I think the main problem is…"

[00:15.2 - 00:17.8]
Speaker 1 + Speaker 3
OVERLAP
[OVERLAPPING SPEECH — TRANSCRIPTION UNCERTAIN]
```

**ASR attribution is a downstream heuristic and is presented as such.** It is never shown as if Nemotron assigned the word. For every word/segment the app retains:

- the raw ASR timestamp
- the Nemotron-active speakers during that interval
- the selected speaker attribution, if any
- attribution confidence
- overlap status

When multiple speakers are active and attribution is ambiguous, the app shows `OVERLAPPING SPEECH — TRANSCRIPTION UNCERTAIN` rather than forcing the word onto one speaker. Speakers are never silently merged.

### Overlap analysis
- Overlap timeline listing every period where two or more channels are active.
- Total recording duration, total speech duration, total overlap duration, and percentage of speech involving overlap.
- Per-speaker statistics: speaking time, turns, average and longest turn, interruptions (where inferable), overlap duration, and share of total speech.

### Timeline UI
- One row per speaker (`Speaker 0` … `Speaker 7`), with active periods as blocks and overlaps made visually obvious.
- Clicking a block jumps the audio/video player to that time, shows the transcript and speaker ID, and lets you play the exact segment.
- For video, the original video is shown next to the timeline. **Speakers are not mapped to faces automatically.**

### Manual annotation
- Optionally map `Speaker N → Person M` by watching a segment (e.g. `Speaker 0 → Person 4`).
- These mappings are stored **separately** from the raw Nemotron output.

---

## Ground-truth benchmark mode

Create ground truth by annotating speaker intervals in the UI, or import an **RTTM** file. Nemotron is then scored against it using standard diarization methodology.

| Metric | Notes |
|---|---|
| **DER** | Diarization Error Rate |
| **MISS** | Missed speech |
| **FA** | False alarm |
| **CONF** | Speaker confusion |
| **Speaker counting accuracy** | 1 if predicted count equals ground truth, else 0 |
| **Speaker-count MAE** | Magnitude of the counting error |

Overlap handling is an explicit choice: **overlap included** or **overlap ignored**. Collar is configurable. No single unlabeled "accuracy" number is reported.

> **Reference labels matter.** Changing the reference RTTM changes the measured result. Nemotron's published numbers use forced-alignment reference labels for AMI, AliMeeting, and NOTSOFAR1, since segment-level transcription annotations can label within-segment silence as speech and inflate MISS. When comparing to published results, use the same references, splits, and collar/overlap settings.

---

## Diagnostic dashboard

The dashboard answers, with measurable evidence and clickable timestamps:

1. How many speakers did Nemotron detect, versus how many were present?
2. Was the count underestimated or overestimated?
3. How much speech was missed, falsely detected, or confused?
4. How much overlap existed?
5. Did speaker labels stay stable across chunks?
6. Which timestamps look suspicious?

Suspicious-case flags include: a speaker disappearing and reappearing as a new ID, two known speakers collapsing into one channel, excessive speaker switching, long false-positive regions, large missed-speech regions, more than 8 apparent acoustic speakers, heavy overlap, and very short segments.

### 1. Speaker-collapse analysis
Computed from the raw activity matrix, for every pair of output channels:

- temporal co-activation and simultaneous activity
- mutually exclusive activity
- activity correlation
- alternating-turn frequency
- total speech duration and overlap duration

With RTTM ground truth, it also maps ground-truth speakers to Nemotron channels and flags cases such as:

```
POSSIBLE SPEAKER COLLAPSE:
Person 3 → 61% Speaker 2
Person 4 → 57% Speaker 2
Person 3 and Person 4 appear to share Speaker 2.
```

The analysis distinguishes **true acoustic collapse**, **postprocessing collapse**, **overlap-related ambiguity**, and **speaker-capacity limitation**. The raw output is never modified.

### 2. Streaming chunk-boundary identity diagnostics
At every streaming chunk boundary the app records the chunk index, boundary timestamp, active speakers before and after, activity-vector similarity where meaningful, and a continuity assessment. Example:

```
Chunk 31 (01:02.0–01:04.0)
Before: Speaker 3   After: Speaker 6
POSSIBLE SPEAKER IDENTITY DISCONTINUITY
```

A speaker being silent around a boundary is **not** flagged on its own; flags require activity/context evidence. This runs per boundary, not only as an offline-vs-streaming comparison.

### 3. Speaker-count funnel

```
MARKED PEOPLE
  ↓
ACTUAL PARTICIPANTS
  ↓
AUDIBLE PARTICIPANTS
  ↓
MAX SIMULTANEOUS ACOUSTIC SPEAKERS
  ↓
NEMOTRON OUTPUT CHANNELS
  ↓
POSTPROCESSED SPEAKER CHANNELS
  ↓
USABLE PROFILES
```

You enter the marked, present, audible, and known-maximum-simultaneous counts manually. Marked people are never equated with the expected Nemotron speaker count.

### 4. Maximum simultaneous speakers
From ground truth: maximum and mean simultaneous speakers during speech, the full distribution, and the percentage of speech with 1, 2, 3, … speakers active.

```
Maximum simultaneous speakers: 10
Nemotron capacity: 8
WARNING: speaker capacity may affect interpretation
```

### 5. Pairwise ground-truth ↔ Nemotron mapping

| Ground truth | Primary Nemotron ID | Coverage | Secondary IDs |
|---|---|---|---|
| Person 1 | Speaker 3 | 87% | Speaker 6 |
| Person 2 | Speaker 1 | 91% | — |
| Person 3 | Speaker 3 | 64% | Speaker 5 |

and the inverse (Nemotron → primary ground truth, with coverage).

### 6. Failure-attribution summary
There is deliberately **no single verdict or quality score**. Instead, each layer is classified independently, and each classification links to its supporting evidence and timestamps:

| Layer | Possible values |
|---|---|
| Preprocessing | PASS / SUSPICIOUS |
| Streaming state | PASS / SUSPICIOUS |
| Speaker capacity | PASS / LIMITATION POSSIBLE |
| Overlap | LOW / MODERATE / HIGH |
| Speaker collapse | NOT DETECTED / POSSIBLE / STRONG EVIDENCE |
| Speaker ID instability | NOT DETECTED / POSSIBLE / STRONG EVIDENCE |
| ASR | INDEPENDENT ISSUE / CONSISTENT / UNCERTAIN |
| Postprocessing | NO EVIDENCE / POSSIBLE ISSUE |

Something is not called a failure merely because it looks unusual.

---

## Controlled test suite

Run the same diagnostic report on each of these to separate model behaviour from data conditions:

| Test | Setup |
|---|---|
| 1 | 2 speakers, clean room, no overlap |
| 2 | 4 speakers, clean room, no overlap |
| 3 | 8 speakers, clean room, no overlap |
| 4 | 4 speakers with interruptions |
| 5 | 4 speakers with deliberate overlap |
| 6 | 8 speakers with overlap |
| 7 | Actual PSYCON recordings |

---

## PSYCON diagnostic report

Export `PSYCON_NEMOTRON_DIAGNOSTIC_REPORT.md`. For every recording it contains: recording ID, marked people, actual participants, estimated audible speakers, maximum simultaneous speakers, Nemotron channels used, speaker-collapse evidence, speaker-ID instability, overlap, MISS, FA, CONF, DER, and observations on ASR, preprocessing, streaming, capacity, and suspicious timestamps.

Followed by an aggregate table:

| Recording | Marked | Audible | Max simultaneous | Nemotron channels | MISS | FA | CONF | DER | Overlap | Collapse | ID instability | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

The report is descriptive and evidence-based. It does not produce a composite Nemotron quality score.

---

## Nemotron reference configurations

Streaming parameters are measured in **80 ms frames**. Latency is `(CHUNK_LEN + RIGHT_CONTEXT) × 80 ms` (input buffer latency, excluding compute). Values below are the model card's recommended configurations.

| Configuration | Latency | `SPKCACHE_LEN` | `FIFO_LEN` | `CHUNK_LEN` | `RIGHT_CONTEXT` | `UPDATE_PERIOD` |
|---|---|---|---|---|---|---|
| Very high latency (offline) | 30.4 s | 264 | 40 | 340 | 40 | 300 |
| Low latency | 1.04 s | 264 | 264 | 9 | 4 | 222 |
| Very low latency | 0.64 s | 264 | 264 | 6 | 2 | 222 |
| Ultra-low latency | 0.32 s | 264 | 264 | 3 | 1 | 222 |

Key model facts (from the [model card](https://huggingface.co/nvidia/Nemotron-3-Diarization)):

- ~100M-parameter Transformer, up to **8 speakers**
- Speaker channels are ordered by **first arrival** in the audio
- Output is a `[T, 8]` tensor of per-speaker activity probabilities; default 10 ms frame stride, configurable in multiples of 10 ms
- Streaming uses an Arrival-Order Speaker Cache (AOSC) and a FIFO queue to keep speaker identity across chunks
- Input: 16 kHz mono audio
- License: [OpenMDW 1.1](https://openmdw.ai/license/1-1/)

---

## Getting started

### Requirements

- Linux (the officially preferred OS for the model)
- NVIDIA GPU with CUDA (the app reports clearly if none is available)
- Python 3.12+
- FFmpeg and `libsndfile1`
- A Hugging Face token if loading the model with `from_pretrained`

### Installation

```bash
git clone https://github.com/CodeSakshamY/PSYCON-Diarization_Model.git
cd PSYCON-Diarization_Model

apt-get update && apt-get install -y libsndfile1 ffmpeg

python -m venv venv
source venv/bin/activate

uv pip install Cython packaging
uv pip install 'nemo-toolkit[asr]'
# TODO: install the app's own dependencies
# pip install -r requirements.txt
```

### Run the app

```bash
# TODO: replace with the actual launch command (Gradio or Streamlit)
python app.py
```

### Workflow

1. **Upload recording** (video or audio).
2. **Run Nemotron** (choose offline or streaming, adjust parameters).
3. Inspect **video player + speaker timeline**.
4. Review the **speaker-attributed transcript**.
5. Review **diagnostic metrics** and the failure-attribution summary.
6. (Optional) Import RTTM or annotate ground truth, then re-score.
7. Download **JSON, RTTM, CSV, or transcript**.

---

## Reproducibility

Every run saves everything needed to reproduce it:

- input filename and hash
- preprocessing parameters
- model identifier and revision (if available)
- all inference parameters
- software versions
- GPU/CPU information
- timestamp
- raw diarization output
- processed RTTM
- transcript
- metrics

## Repository structure

```
PSYCON-Diarization_Model/
├── nemotron_bench/              # Benchmark application code
├── nemotron_bench.zip           # Zipped copy of the benchmark package
├── nemotron_bench_colab.ipynb   # Colab notebook
└── README.md
```

`TODO:` Update with module-level layout (preprocessing, inference, metrics, diagnostics, UI) once finalized.

## Design principles

- **Raw output is immutable.** Postprocessing, ASR attribution, and manual annotation are stored separately.
- **No silent substitutions.** No alternate diarization model, no silent CPU fallback.
- **Evidence over verdicts.** Diagnostics are descriptive, measurable, and linked to timestamps.
- **Layers stay separable.** Preprocessing, inference, streaming state, overlap, capacity, ASR, and downstream processing can each be examined on their own.
- **No face-to-speaker automation.** Speaker-to-person mapping is manual only.

## References

- [Nemotron-3-Diarization model card](https://huggingface.co/nvidia/Nemotron-3-Diarization)
- [Sortformer](https://arxiv.org/abs/2409.06656)
- [Streaming Sortformer](https://arxiv.org/abs/2507.18446)
- [Can We Really Repurpose Multi-Speaker ASR Corpus for Speaker Diarization?](https://arxiv.org/abs/2507.09226) (forced-alignment reference labels)
- [NVIDIA NeMo Speech](https://github.com/NVIDIA-NeMo/Speech)

## License

`TODO:` Add a license for this repository. Use of the Nemotron model itself is governed by the OpenMDW License 1.1.

## Contact

Maintained by [@CodeSakshamY](https://github.com/CodeSakshamY).
