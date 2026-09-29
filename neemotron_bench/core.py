"""Independent Nemotron-3-Diarization benchmark core. No other diarizer is ever used."""
import csv, datetime, hashlib, json, pathlib, platform, re, shutil, subprocess, time
from importlib import metadata
import numpy as np

MODEL = "nvidia/Nemotron-3-Diarization"
RUNS = pathlib.Path("static/runs")  # served by Streamlit static serving
NSPK = 8
# From the model card (all lengths in 80 ms frames)
PRESETS = {
    "Offline (30.4 s)": dict(spkcache_len=264, fifo_len=40, chunk_len=340, chunk_right_context=40, spkcache_update_period=300),
    "Streaming 1.04 s": dict(spkcache_len=264, fifo_len=264, chunk_len=9, chunk_right_context=4, spkcache_update_period=222),
    "Streaming 0.64 s": dict(spkcache_len=264, fifo_len=264, chunk_len=6, chunk_right_context=2, spkcache_update_period=222),
    "Streaming 0.32 s": dict(spkcache_len=264, fifo_len=264, chunk_len=3, chunk_right_context=1, spkcache_update_period=222),
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def preprocess(src, out):
    """Explicit mono / 16 kHz / 16-bit PCM WAV; returns logged audio properties."""
    cmd = ["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out)]
    subprocess.run(cmd, check=True, capture_output=True)
    pr = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(out)],
                                   capture_output=True, text=True).stdout)
    s = pr["streams"][0]
    sd = float(json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_format", "-of", "json", str(src)], capture_output=True, text=True).stdout)["format"]["duration"])
    return dict(src_duration_s=sd, ffmpeg_cmd=" ".join(cmd), codec=s["codec_name"], sample_rate=int(s["sample_rate"]),
                channels=int(s["channels"]), duration_s=float(pr["format"]["duration"]))


def levels(wav):
    import wave
    with wave.open(str(wav)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    return dict(peak=float(np.abs(x).max()), rms_dbfs=float(20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-9)), clip_frac=float((np.abs(x) >= 0.999).mean()))


def runs(mask):
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def run_nemotron(wav, cfg, allow_cpu=False):
    import torch
    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError("CUDA is not available. Refusing to continue silently (tick 'allow CPU' to run the same model on CPU).")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from nemo.collections.asr.models import SortformerEncLabelModel
    m = SortformerEncLabelModel.from_pretrained(MODEL).to(dev).eval()
    for k, v in cfg.items():
        setattr(m.sortformer_modules, k, v)
    m._check_streaming_parameters()  # NeMo's streaming loop carries speaker cache + FIFO across chunks
    t = time.time()
    segs, probs = m.diarize(audio=[str(wav)], batch_size=1, include_tensor_outputs=True)
    secs = time.time() - t
    p = probs[0]
    p = np.asarray(p.float().cpu() if hasattr(p, "float") else p).squeeze()
    if p.shape[-1] != NSPK: p = p.T  # want [T, 8]
    raw = [str(s) for s in segs[0]]
    hw = dict(device=dev, gpu=torch.cuda.get_device_name(0) if dev == "cuda" else None, cpu=platform.processor(), platform=platform.platform())
    return p, raw, secs, hw


def probs_to_segments(p, fs, thr):
    act = p >= thr
    out = []
    for s in range(p.shape[1]):
        for a, b in runs(act[:, s]):
            out.append(dict(speaker=s, start=a * fs, end=b * fs, duration=(b - a) * fs, confidence=float(p[a:b, s].mean())))
    return sorted(out, key=lambda x: (x["start"], x["speaker"])), act


def analyze(act, fs, segs, min_speech=1.0):
    n = act.sum(1); ov = n >= 2; sp = n >= 1
    speech, ovl = sp.sum() * fs, ov.sum() * fs
    ovp = [dict(start=a * fs, end=b * fs, duration=(b - a) * fs, speakers=[int(x) for x in np.where(act[a:b].any(0))[0]]) for a, b in runs(ov)]
    stats = []
    for s in range(act.shape[1]):
        mine = [g for g in segs if g["speaker"] == s]
        if not mine: continue
        tot = act[:, s].sum() * fs
        intr = sum(1 for g in mine if np.delete(act[max(int(g["start"] / fs) - 1, 0)], s).any())
        stats.append({"speaker": f"Speaker {s}", "total_s": round(tot, 2), "turns": len(mine), "avg_turn_s": round(tot / len(mine), 2),
                      "longest_s": round(max(g["duration"] for g in mine), 2), "interruptions": intr,
                      "overlap_s": round((act[:, s] & ov).sum() * fs, 2), "pct_of_speech": round(100 * tot / max(speech, 1e-9), 1)})
    detected = [s for s in stats if s["total_s"] >= min_speech]
    flags = []
    for s in range(act.shape[1]):
        ss = sorted([g for g in segs if g["speaker"] == s], key=lambda g: g["start"])
        for a, b in zip(ss, ss[1:]):
            if b["start"] - a["end"] > 60:
                flags.append(f"Speaker {s} silent {b['start']-a['end']:.0f}s then reappears at {b['start']:.1f}s (possible identity drop / re-ID)")
    for g in segs:
        if g["duration"] > 45: flags.append(f"Long single-speaker region {g['start']:.1f}-{g['end']:.1f}s (Speaker {g['speaker']}) - possible false alarm")
    if segs and sum(g["duration"] < 0.3 for g in segs) / len(segs) > 0.2: flags.append("Over 20% of segments are shorter than 0.3 s (fragmented output)")
    order = [g["speaker"] for g in sorted(segs, key=lambda g: g["start"])]
    if segs and sum(a != b for a, b in zip(order, order[1:])) / max(len(act) * fs / 60, 1e-9) > 30: flags.append("Excessive speaker switching (>30 changes/min)")
    if len(detected) >= NSPK: flags.append("All 8 channels used: model cannot represent >8 speakers, extra speakers may collapse")
    if speech and 100 * ovl / speech > 20: flags.append(f"Heavy overlap ({100*ovl/speech:.0f}% of speech)")
    for o in sorted(ovp, key=lambda o: -o["duration"])[:3]: flags.append(f"Longest overlaps: {o['start']:.1f}-{o['end']:.1f}s speakers {o['speakers']}")
    return dict(summary=dict(duration_s=round(len(act) * fs, 2), speech_s=round(speech, 2), overlap_s=round(ovl, 2),
                             overlap_pct_of_speech=round(100 * ovl / max(speech, 1e-9), 1), detected_speakers=len(detected)),
                speaker_stats=stats, overlap_timeline=ovp, flags=flags)


def read_rttm(path):
    return [(float(f[3]), float(f[3]) + float(f[4]), f[7]) for f in (l.split() for l in open(path)) if f and f[0] == "SPEAKER"]


def write_rttm(path, segs, name):
    with open(path, "w") as f:
        for g in segs:
            f.write(f"SPEAKER {name} 1 {g['start']:.3f} {g['duration']:.3f} <NA> <NA> speaker_{g['speaker']} <NA> <NA>\n")


def der(ref_rttm, segs, dur, collar=0.0, skip_overlap=False):
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate
    ref, hyp = Annotation(), Annotation()
    for a, b, s in read_rttm(ref_rttm): ref[Segment(a, b)] = s
    for g in segs: hyp[Segment(g["start"], g["end"])] = f"Speaker {g['speaker']}"
    d = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)(ref, hyp, uem=Segment(0, dur), detailed=True)
    t = max(d["total"], 1e-9); nr, nh = len(ref.labels()), len(hyp.labels())
    return dict(DER=d["diarization error rate"], MISS=d["missed detection"] / t, FA=d["false alarm"] / t, CONF=d["confusion"] / t,
                ref_speech_s=d["total"], collar=collar, overlap_ignored=skip_overlap, true_speakers=nr, hyp_speakers=nh,
                speaker_count_correct=int(nr == nh), speaker_count_abs_err=abs(nr - nh))


def transcribe(wav, size="large-v3", lang=None, dev="cuda"):
    from faster_whisper import WhisperModel
    w = WhisperModel(size, device=dev, compute_type="float16" if dev == "cuda" else "int8")
    segs, _ = w.transcribe(str(wav), word_timestamps=True, language=lang, vad_filter=False)
    return [dict(w=x.word.strip(), start=x.start, end=x.end, p=x.probability) for s in segs for x in s.words]


def attribute(words, act, fs, gap=1.5, act_frac=0.2):
    """DOWNSTREAM HEURISTIC, not Nemotron output. A word is attributed to a speaker only if exactly one channel
    is active (>= act_frac of the word's frames); 2+ active channels => overlap/uncertain. Raw diarization is untouched."""
    ws = []
    for w in words:
        a = int(w["start"] / fs); b = min(max(int(w["end"] / fs), a + 1), len(act)); fr = act[a:b].mean(0)
        active = [int(s) for s in np.where(fr >= act_frac)[0]]
        sel = active[0] if len(active) == 1 else None
        ws.append(dict(asr_word=w["w"], asr_start=w["start"], asr_end=w["end"], asr_prob=w["p"],
                       nemotron_active_fraction={f"Speaker {s}": round(float(fr[s]), 2) for s in np.where(fr > 0)[0]},
                       active_speakers=active, selected_speaker=sel,
                       attribution_confidence=round(float(fr[sel] - np.delete(fr, sel).max()), 2) if sel is not None else None,
                       status="overlap" if len(active) >= 2 else "single" if sel is not None else "no_active_speaker"))
    groups = []
    for w in ws:
        t = tuple(w["active_speakers"])
        if groups and groups[-1]["speakers"] == t and w["asr_start"] - groups[-1]["end"] < gap:
            g = groups[-1]; g["end"] = w["asr_end"]; g["_w"].append(w)
        else:
            groups.append(dict(speakers=t, start=w["asr_start"], end=w["asr_end"], _w=[w]))
    for g in groups:
        w = g.pop("_w"); g["overlap"] = len(g["speakers"]) >= 2
        g["text"] = " ".join(x["asr_word"] for x in w); g["confidence"] = float(np.mean([x["asr_prob"] for x in w]))
        cs = [x["attribution_confidence"] for x in w if x["attribution_confidence"] is not None]
        g["attribution_confidence"] = float(np.mean(cs)) if cs else None; g["attribution"] = "heuristic (ASR timing x Nemotron activity)"
    return groups, ws


def fmt(t): return f"{int(t//60):02d}:{t%60:06.3f}"


def transcript_text(groups):
    out = []
    for g in groups:
        if g["overlap"]:
            lab = " + ".join(f"Speaker {s}" for s in g["speakers"]) + "\nOVERLAP"
            body = f"[OVERLAPPING SPEECH - TRANSCRIPTION UNCERTAIN]\n(raw ASR guess, not attributable: {g['text']})"
        elif g["speakers"]:
            lab = f"Speaker {g['speakers'][0]} (heuristic attribution, conf {g['attribution_confidence']:.2f})"; body = f"\"{g['text']}\""
        else:
            lab = "No Nemotron speaker active (unattributed)"; body = f"\"{g['text']}\""
        out.append(f"[{fmt(g['start'])} - {fmt(g['end'])}]\n{lab}:\n{body}  (ASR conf {g['confidence']:.2f})\n")
    return "\n".join(out)


def ver(p):
    try: return metadata.version(p)
    except Exception: return None


def run_all(src, preset, cfg, thr, asr=True, asr_size="large-v3", lang=None, allow_cpu=False, tag="", gt_rttm=None):
    src = pathlib.Path(src)
    rd = RUNS / f"{datetime.datetime.now():%Y%m%d_%H%M%S}_{src.stem}"
    rd.mkdir(parents=True)
    media = rd / f"media{src.suffix.lower()}"; shutil.copy(src, media)
    wav = rd / "audio_16k_mono.wav"
    pre = preprocess(src, wav); pre.update(levels(wav))
    p, raw, secs, hw = run_nemotron(wav, cfg, allow_cpu)
    fs = pre["duration_s"] / len(p)
    segs, act = probs_to_segments(p, fs, thr)
    np.save(rd / "raw_probs.npy", p)
    json.dump(raw, open(rd / "raw_nemo_segments.json", "w"), indent=1)
    with open(rd / "segments.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["speaker", "start", "end", "duration", "confidence"]); w.writeheader(); w.writerows(segs)
    write_rttm(rd / "hyp.rttm", segs, src.stem)
    an = analyze(act, fs, segs)
    groups, words = attribute(transcribe(wav, asr_size, lang, hw["device"]), act, fs) if asr else ([], [])
    json.dump(words, open(rd / "words.json", "w"), indent=1)
    (rd / "transcript.txt").write_text(transcript_text(groups)); json.dump(groups, open(rd / "transcript.json", "w"), indent=1)
    metrics = None
    if gt_rttm:
        shutil.copy(gt_rttm, rd / "ref.rttm")
    manifest = dict(input=src.name, sha256=sha256(src), tag=tag, timestamp=datetime.datetime.now().isoformat(), preprocessing=pre,
                    model=MODEL, model_revision=_rev(), preset=preset, inference_params=cfg, activity_threshold=thr,
                    frame_resolution_s=fs, n_frames=len(p), inference_seconds=secs, rtfx=pre["duration_s"] / secs, hardware=hw,
                    asr=dict(enabled=asr, model=asr_size, lang=lang),
                    versions={k: ver(k) for k in ["torch", "nemo_toolkit", "faster-whisper", "pyannote.metrics", "streamlit"]},
                    ffmpeg=subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0])
    json.dump(dict(manifest=manifest, analysis=an, segments=segs), open(rd / "run.json", "w"), indent=1)
    return rd


def _rev():
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(MODEL).sha
    except Exception:
        return None
