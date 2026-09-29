# Nemotron-3-Diarization benchmark: manual test guide

Independent of the PSYCON pipeline. Only `nvidia/Nemotron-3-Diarization` is used for diarization.

## 0. Setup

Requirements: Linux, Python 3.12+, NVIDIA GPU with CUDA PyTorch, `ffmpeg`/`ffprobe` on PATH, a Hugging Face login (`huggingface-cli login`) if the model download asks for it.

```bash
cd nemotron_bench
pip install -r requirements.txt
streamlit run app.py
```

Run it from this folder: outputs go to `./static/runs/<timestamp>_<name>/` and are served through Streamlit static serving (`.streamlit/config.toml`).

## 1. Smoke tests (do these first, ~10 min)

These check that the tool works before you trust it on PSYCON.

**1a. Environment**
- App opens with no import errors. If NeMo fails to import, fix that first.
- With no GPU, RUN NEMOTRON must show a CUDA error. Nothing should run silently. Tick "Allow CPU" only if you want a slow run of the same model.

**1b. Any short multi-speaker clip (1-3 min), offline preset**
- Upload, keep "Offline (30.4 s)", threshold 0.5, ASR on, click RUN NEMOTRON.
- Expected: video/audio player, timeline with coloured blocks, transcript, diagnostics.
- Open the run folder and confirm these files exist: `audio_16k_mono.wav`, `raw_nemo_segments.json`, `raw_probs.npy`, `segments.csv`, `hyp.rttm`, `words.json`, `transcript.txt`, `run.json`, `diagnosis.json`.
- In "Run manifest", confirm: sample rate 16000, channels 1, codec `pcm_s16le`, model revision filled, GPU name, inference seconds.

**1c. Check the raw tensor shape** (this was written defensively and not verified against NeMo)
```bash
python -c "import numpy as np; p=np.load('static/runs/<run>/raw_probs.npy'); print(p.shape, p.min(), p.max())"
```
Expected `(frames, 8)` with values in [0, 1]. `frames` x `frame_resolution_s` in the manifest should equal the audio duration. If not, stop and report it.

**1d. Timeline clicking**
- Click a block: player jumps there, plays that segment only, and the info line shows speaker, times and ASR text.
- Click a timestamp button under FAILURE ATTRIBUTION: the player jumps to it.
- If a MOV/odd-codec video does not play, that is a browser limit. Analysis still works. Re-encode with `ffmpeg -i in.mov -c:v libx264 -c:a aac out.mp4` if you want playback.

**1e. Metric self-check (DER must be 0)**
- Upload the run's own `hyp.rttm` in "Reference RTTM", collar 0, overlap "Included".
- Expected: DER 0.0, MISS 0, FA 0, CONF 0, speaker counts equal. If not, the evaluation path is broken.

## 2. Build controlled test recordings with exact ground truth

Recorded real conversations are hard to label precisely. For TESTS 1-6 it is easier to build the audio yourself from single-speaker clips (one clean file per person, e.g. 20-60 s each, different voices), so the ground truth is known exactly.

**Sequential, no overlap** (A 0-10 s, then B, then C):
```bash
ffmpeg -i A.wav -i B.wav -i C.wav -filter_complex "[0][1][2]concat=n=3:v=0:a=1" seq.wav
```

**Deliberate overlap** (B starts 5 s into A):
```bash
ffmpeg -i A.wav -i B.wav -filter_complex "[1]adelay=5000|5000[b];[0][b]amix=inputs=2:duration=longest:normalize=0" mix.wav
```
(`normalize=0` needs ffmpeg 4.4+. Otherwise drop it and accept a level drop.)

**Write the RTTM from your known timings.** One line per turn, times in seconds:
```
SPEAKER seq 1 0.000 10.000 <NA> <NA> Person_1 <NA> <NA>
SPEAKER seq 1 10.000 12.500 <NA> <NA> Person_2 <NA> <NA>
```
Fields: `SPEAKER <file> 1 <start> <duration> <NA> <NA> <person> <NA> <NA>`. Use each clip's real speech extent (trim leading/trailing silence, or use the clip's actual duration if it is dense speech). Overlapping turns are just overlapping lines.

## 3. The test suite (spec TESTS 1-7)

For each test: pick the "Test tag" in the sidebar, upload, RUN NEMOTRON, upload the RTTM under Ground truth, then fill the funnel boxes (marked / present / audible / known max simultaneous).

| Test | Recording | What to look for |
|---|---|---|
| 1 | 2 speakers, clean, no overlap | Baseline. Expect 2 channels, low DER, no collapse, no boundary flags. If this is bad, suspect setup/preprocessing, not the model. |
| 2 | 4 speakers, clean, no overlap | Expect 4 channels and a clean GT → Nemotron mapping (one primary channel per person, coverage high). |
| 3 | 8 speakers, clean, no overlap | Channel count near 8 and capacity flag "LIMITATION POSSIBLE" only because all 8 channels are used. Check for collapse. |
| 4 | 4 speakers with interruptions | Overlap should be LOW/MODERATE. Check the interruption column and short segments. |
| 5 | 4 speakers, deliberate overlap | Overlap timeline should line up with where you overlapped. Words in those regions must show OVERLAPPING SPEECH — TRANSCRIPTION UNCERTAIN. |
| 6 | 8 speakers with overlap | Stress test: collapse, ID instability, capacity, and DER split into MISS/FA/CONF. |
| 7 | PSYCON recordings | Same report. Enter marked/present/audible honestly. Do not set "marked" as the expected channel count. |

**How to read a result**
1. FAILURE ATTRIBUTION: each item is separate, with evidence and clickable times. Do not average them into a verdict.
2. GT → Nemotron mapping: two people sharing one primary channel is collapse; one person split across channels is fragmentation.
3. DER: compute both ways: overlap "Included" and "Ignored", with your chosen collar. Report which you used. If DER is high only with overlap included, the problem is overlap, not general diarization.

## 4. Isolating the cause (the actual PSYCON question)

Run the **same file** several times and compare in the comparison table at the bottom of the app:

1. **Offline vs streaming:** Offline (30.4 s) vs Streaming 1.04 s vs 0.32 s. A big DER or channel-count jump in streaming → suspect streaming state/config.
2. **Threshold sweep:** re-run at 0.3, 0.5, 0.7. The POSTPROCESSING item also shows this from one run. If channel count or speech time swings, the channel splitting is threshold-sensitive.
3. **Preprocessing:** confirm PREPROCESSING is PASS. Compare your existing PSYCON audio pipeline's WAV against `audio_16k_mono.wav` (sample rate, level, clipping).
4. **Capacity:** if known max simultaneous speakers is over 8, or all 8 channels are used with more than 8 participants, interpret collapse with that in mind.
5. **Chunk boundaries:** for streaming runs, open the STREAMING STATE item and click the flagged timestamps. Listen: a flag on a true speaker change is a false alarm; a flag mid-sentence by one person is a real identity discontinuity.
6. **ASR:** ASR issues show as low word confidence with little overlap. This is independent of diarization.

Then click "Generate PSYCON_NEMOTRON_DIAGNOSTIC_REPORT.md" (only runs tagged `TEST 7` are included) and read the aggregate table.

## 5. Known limits to keep in mind

- Speaker labels are anonymous and ordered by first appearance. Speaker 0 is not Person 1. Use the mapping table or the manual mapping editor (saved separately in `person_map.json`).
- Chunk-boundary times assume uniform chunks of `chunk_len` x 80 ms. Verify a few by ear.
- Collapse "type" labels are heuristics with the numbers shown. Check the evidence lines.
- ASR speaker attribution is a downstream heuristic, not Nemotron output. Per-word details are in `words.json`.
- Ground truth is imported as RTTM only. There is no in-app interval editor.
- "Usable profiles" is a researcher-defined threshold (minimum speech seconds), not something Nemotron provides.

## 6. Reproducibility check

Re-run the same file with the same settings and compare `raw_probs.npy` between runs (`np.allclose`). Small numeric differences are possible on GPU. Segment counts and channel usage should match. Each run's `run.json` records the input SHA-256, model revision, parameters, versions and hardware.
