"""Derived diagnostics. Reads raw_probs.npy / run.json / ref.rttm and writes ONLY diagnosis.json. Raw outputs are never modified."""
import json, pathlib
import numpy as np
import core

CAP = core.NSPK


def gt_matrix(rttm, n, fs):
    ref = core.read_rttm(rttm); names = sorted({s for _, _, s in ref})
    M = np.zeros((len(names), n), bool)
    for a, b, s in ref: M[names.index(s), int(a / fs):min(int(np.ceil(b / fs)), n)] = True
    return names, M


def dist(M):  # M: [speakers, frames]
    c = M.sum(0); sp = c[c >= 1]
    return dict(max=int(c.max()) if c.size else 0, mean_during_speech=float(sp.mean()) if sp.size else 0.0,
                pct_of_speech={int(k): round(100 * float((sp == k).mean()), 1) for k in range(1, int(sp.max()) + 1)} if sp.size else {})


def pairwise(act, fs, segs):
    used = [s for s in range(act.shape[1]) if act[:, s].any()]; rows = []
    for i in used:
        for j in used:
            if j <= i: continue
            a, b = act[:, i], act[:, j]; both, either = (a & b).sum(), (a | b).sum()
            ss = sorted([g for g in segs if g["speaker"] in (i, j)], key=lambda g: g["start"])
            alt = sum(1 for x, y in zip(ss, ss[1:]) if x["speaker"] != y["speaker"] and 0 <= y["start"] - x["end"] <= 2)
            r = np.nan_to_num(np.corrcoef(a, b)[0, 1])
            rows.append({"pair": f"Speaker {i} / Speaker {j}", "total_s_a": round(a.sum() * fs, 1), "total_s_b": round(b.sum() * fs, 1),
                         "simultaneous_s (=pair overlap)": round(both * fs, 2), "coactivation (Jaccard)": round(both / max(either, 1), 3),
                         "exclusivity": round(1 - both / max(min(a.sum(), b.sum()), 1), 3), "correlation": round(float(r), 3), "alternations<=2s": alt})
    return rows


def mapping(act, names, G, fs):
    C = np.array([[(G[i] & act[:, j]).sum() * fs for j in range(act.shape[1])] for i in range(len(names))])
    gt_tot, hy_tot = G.sum(1) * fs, act.sum(0) * fs; fwd, inv = [], []
    for i, n in enumerate(names):
        if gt_tot[i] == 0: continue
        cov = C[i] / gt_tot[i]; p = int(cov.argmax())
        fwd.append({"Ground Truth": n, "Primary Nemotron ID": f"Speaker {p}", "Coverage %": round(100 * cov[p]),
                    "Secondary IDs": ", ".join(f"Speaker {j} ({100*cov[j]:.0f}%)" for j in np.argsort(-cov) if j != p and cov[j] >= 0.15) or "-"})
    for j in range(act.shape[1]):
        if hy_tot[j] == 0: continue
        cov = C[:, j] / hy_tot[j]; p = int(cov.argmax())
        inv.append({"Nemotron": f"Speaker {j}", "Primary Ground Truth": names[p], "Coverage %": round(100 * cov[p])})
    return C, fwd, inv


def collapse(act, p, thr, names, G, C, fs):
    cov = C / np.maximum(G.sum(1)[:, None] * fs, 1e-9); out = []
    for j in range(act.shape[1]):
        P = [i for i in range(len(names)) if cov[i].argmax() == j and cov[i, j] >= 0.35]
        if len(P) < 2: continue
        types, ev = [], [f"{names[i]} -> {100*cov[i,j]:.0f}% Speaker {j}" for i in P]
        u = G[P]; ogt = float((u.sum(0) >= 2).sum() / max((u.sum(0) >= 1).sum(), 1))
        if ogt >= 0.3: types.append("OVERLAP-RELATED AMBIGUITY"); ev.append(f"{100*ogt:.0f}% of these people's combined speech is simultaneous in ground truth")
        oth = np.delete(p, j, axis=1).max(1); sub = (oth >= 0.5 * thr) & (oth < thr)
        fr = max(float(sub[G[i]].mean()) for i in P)
        if fr >= 0.25: types.append("POSTPROCESSING COLLAPSE (threshold)"); ev.append(f"a competing channel sits at 0.5-1.0x threshold in {100*fr:.0f}% of a person's frames; lower the threshold to test")
        if G.sum(0).max() > CAP or (len(names) > CAP and act.any(0).sum() == CAP): types.append("SPEAKER-CAPACITY LIMITATION"); ev.append(f"GT max simultaneous {int(G.sum(0).max())}, participants {len(names)}, capacity {CAP}")
        if not types: types.append("TRUE ACOUSTIC / MODEL-LEVEL COLLAPSE"); ev.append("no threshold, overlap or capacity explanation found (heuristic)")
        out.append(dict(channel=j, persons=[names[i] for i in P], types=types, evidence=ev, strength="STRONG" if min(cov[i, j] for i in P) >= 0.5 else "POSSIBLE",
                        times=[float(x) for x in np.where(act[:, j] & G[P].any(0))[0][:3] * fs]))
    return out


def boundaries(act, fs, chunk_len, G=None, names=None, win=2.0):
    step, W, g = chunk_len * 0.08, max(int(win / fs), 1), max(int(0.3 / fs), 1); n = len(act); rows = []
    for k in range(1, int(n * fs / step) + 1):
        b = int(round(k * step / fs))
        if b >= n: break
        pre, post = act[max(b - W, 0):b], act[b:b + W]; vb, va = pre.mean(0), post.mean(0)
        sb, sa = [int(s) for s in np.where(vb >= 0.1)[0]], [int(s) for s in np.where(va >= 0.1)[0]]
        cos = float(vb @ va / (np.linalg.norm(vb) * np.linalg.norm(va))) if sb and sa else None
        contin = act[max(b - g, 0):b].any() and act[b:b + g].any()
        st = "not assessable (silence/no speech on a side)" if not (sb and sa) else "same channel(s) continue" if set(sb) & set(sa) else "plausible speaker change (pause at boundary)"
        if sb and sa and not set(sb) & set(sa) and contin and len(sb) == 1 and len(sa) == 1:
            st = "POSSIBLE SPEAKER IDENTITY DISCONTINUITY"
            if G is not None:
                pb = {names[i] for i in range(len(names)) if G[i, max(b - W, 0):b].mean() >= 0.3}; pa = {names[i] for i in range(len(names)) if G[i, b:b + W].mean() >= 0.3}
                if len(pb) == 1 and pb == pa: st = "CONFIRMED DISCONTINUITY (same GT person, channel changed)"
                elif pb and pa and pb != pa: st = "plausible speaker change (GT confirms different person)"
        rows.append(dict(chunk=k, boundary_s=round(b * fs, 2), before=sb, after=sa, cosine=None if cos is None else round(cos, 3), continuous_speech=bool(contin), status=st))
    return rows


def item(name, status, ev, times=()): return dict(name=name, status=status, evidence=ev, times=[float(t) for t in list(times)[:8]])


def failures(man, an, d, words, act, fs, fu):
    pre = man["preprocessing"]; F = []; ev = []; bad = False
    if pre["sample_rate"] != 16000 or pre["channels"] != 1 or pre["codec"] != "pcm_s16le": bad = True; ev.append("output not mono/16 kHz/pcm_s16le")
    if abs(pre.get("src_duration_s", pre["duration_s"]) - pre["duration_s"]) > 0.5: bad = True; ev.append("duration changed by conversion")
    if pre.get("clip_frac", 0) > 0.001: bad = True; ev.append(f"clipping in {100*pre['clip_frac']:.2f}% of samples")
    if pre.get("rms_dbfs", 0) < -50: bad = True; ev.append("very low level (RMS < -50 dBFS)")
    F.append(item("PREPROCESSING", "SUSPICIOUS" if bad else "PASS", ev + [f"peak {pre.get('peak',0):.2f}, RMS {pre.get('rms_dbfs',0):.1f} dBFS"]))
    fl = [b for b in d["boundaries"] if b["status"].startswith(("POSSIBLE", "CONFIRMED"))]
    F.append(item("STREAMING STATE", "SUSPICIOUS" if fl else "PASS", [f"{len(fl)}/{len(d['boundaries'])} chunk boundaries flagged (chunk_len={man['inference_params']['chunk_len']} frames)"], [b["boundary_s"] for b in fl]))
    used = int((act.sum(0) > 0).sum()); gmax = d.get("gt_dist", {}).get("max")
    F.append(item("SPEAKER CAPACITY", "LIMITATION POSSIBLE" if used == CAP or (gmax or 0) > CAP else "PASS", [f"channels used {used}/{CAP}", f"GT max simultaneous {gmax}" if gmax else "GT max simultaneous unknown"]))
    op = an["summary"]["overlap_pct_of_speech"]
    F.append(item("OVERLAP", "LOW" if op < 5 else "MODERATE" if op < 20 else "HIGH", [f"{op}% of speech has >=2 active channels (bands: <5 low, <20 moderate)"], [o["start"] for o in sorted(an["overlap_timeline"], key=lambda o: -o["duration"])]))
    col = d.get("collapse")
    if col is None:
        aud = fu.get("audible", 0)
        F.append(item("SPEAKER COLLAPSE", "POSSIBLE" if aud and used < aud else "NOT ASSESSABLE (no ground truth)", [f"channels used {used} vs estimated audible {aud or 'n/a'}"]))
    else:
        F.append(item("SPEAKER COLLAPSE", "STRONG EVIDENCE" if any(c["strength"] == "STRONG" for c in col) else "POSSIBLE" if col else "NOT DETECTED",
                      [f"Speaker {c['channel']}: {', '.join(c['persons'])} [{'; '.join(c['types'])}]" for c in col], [t for c in col for t in c["times"]]))
    conf = [b for b in fl if b["status"].startswith("CONFIRMED")]
    F.append(item("SPEAKER ID INSTABILITY", "STRONG EVIDENCE" if conf or len(fl) >= 3 else "POSSIBLE" if fl else "NOT DETECTED", [f"{len(fl)} flagged boundaries, {len(conf)} GT-confirmed"], [b["boundary_s"] for b in fl]))
    if not words: F.append(item("ASR", "UNCERTAIN", ["ASR disabled or no words"]))
    else:
        n = len(words); low = sum(w["asr_prob"] < 0.5 for w in words) / n; un = [w for w in words if w["status"] == "no_active_speaker"]; ov = sum(w["status"] == "overlap" for w in words) / n
        s = "INDEPENDENT ISSUE" if low >= 0.25 and ov < 0.1 else "CONSISTENT" if low < 0.1 and len(un) / n < 0.05 else "UNCERTAIN"
        F.append(item("ASR", s, [f"{100*low:.0f}% words ASR prob<0.5", f"{100*len(un)/n:.0f}% words with no active Nemotron speaker", f"{100*ov:.0f}% words in overlap"], [w["asr_start"] for w in un]))
    sens = d["threshold_sensitivity"]; sp = [x["speech_s"] for x in sens]; ch = {x["channels"] for x in sens}
    pp = len(ch) > 1 or (sp[1] > 0 and (max(sp) - min(sp)) / sp[1] > 0.15) or any("POSTPROCESSING" in t for c in (col or []) for t in c["types"])
    F.append(item("POSTPROCESSING", "POSSIBLE ISSUE" if pp else "NO EVIDENCE", [f"thr {x['thr']}: {x['channels']} channels, {x['speech_s']} s speech" for x in sens]))
    return F


def diagnose(rd):
    rd = pathlib.Path(rd); run = json.load(open(rd / "run.json")); man, an, segs = run["manifest"], run["analysis"], run["segments"]
    p = np.load(rd / "raw_probs.npy"); thr, fs = man["activity_threshold"], man["frame_resolution_s"]; act = p >= thr
    fu = json.load(open(rd / "funnel.json")) if (rd / "funnel.json").exists() else {}
    words = json.load(open(rd / "words.json")) if (rd / "words.json").exists() else []
    d = dict(pairwise=pairwise(act, fs, segs), simultaneous_nemotron=dist(act.T))
    names = G = None
    if (rd / "ref.rttm").exists():
        names, G = gt_matrix(rd / "ref.rttm", len(p), fs); C, d["forward"], d["inverse"] = mapping(act, names, G, fs)
        d["gt_dist"] = dist(G); d["collapse"] = collapse(act, p, thr, names, G, C, fs)
    d["boundaries"] = boundaries(act, fs, man["inference_params"]["chunk_len"], G, names)
    d["threshold_sensitivity"] = [dict(thr=t, channels=int(((p >= t).sum(0) * fs >= 1).sum()), speech_s=round(float((p >= t).any(1).sum() * fs), 1)) for t in (0.3, 0.5, 0.7)]
    ms = fu.get("max_simul") or d.get("gt_dist", {}).get("max")
    d["funnel"] = [("MARKED PEOPLE", fu.get("marked") or None), ("ACTUAL PARTICIPANTS", fu.get("present") or None), ("AUDIBLE PARTICIPANTS", fu.get("audible") or None),
                   ("MAX SIMULTANEOUS ACOUSTIC SPEAKERS", ms or None), ("NEMOTRON OUTPUT CHANNELS (>=0.5 s)", int((act.sum(0) * fs >= 0.5).sum())),
                   ("POSTPROCESSED SPEAKER CHANNELS (>=1 s)", an["summary"]["detected_speakers"]),
                   (f"USABLE PROFILES (>={fu.get('usable_min_s', 10)} s speech; researcher-defined)", sum(s["total_s"] >= fu.get("usable_min_s", 10) for s in an["speaker_stats"]))]
    d["failure_attribution"] = failures(man, an, d, words, act, fs, fu)
    json.dump(d, open(rd / "diagnosis.json", "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return d


def build_report(root, only_tag="TEST 7"):
    L = ["# PSYCON_NEMOTRON_DIAGNOSTIC_REPORT", "", "Descriptive, evidence-based. No composite quality score. Marked people are NOT expected Nemotron speaker counts.", ""]; rows = []; ch_total = mk_total = 0
    pc = lambda v: "n/a" if v is None else f"{100*v:.1f}%"
    for r in sorted(pathlib.Path(root).glob("*/run.json")):
        rd = r.parent; run = json.load(open(r)); m = run["manifest"]
        if only_tag and not m["tag"].startswith(only_tag): continue
        d = diagnose(rd); fu = json.load(open(rd / "funnel.json")) if (rd / "funnel.json").exists() else {}
        mt = json.load(open(rd / "metrics.json")) if (rd / "metrics.json").exists() else {}
        F = {f["name"]: f for f in d["failure_attribution"]}; f = dict(d["funnel"]); ch = run["analysis"]["summary"]["detected_speakers"]
        ch_total += ch; mk_total += fu.get("marked", 0); sus = sorted({t for x in F.values() if x["status"] not in ("PASS", "NOT DETECTED", "NO EVIDENCE", "LOW") for t in x["times"]})[:10]
        L += [f"## {m['input']}  ({m['preset']}, {m['tag'] or 'untagged'})", f"- Marked people: {fu.get('marked','n/a')} | actual participants: {fu.get('present','n/a')} | est. audible: {fu.get('audible','n/a')} | max simultaneous: {f['MAX SIMULTANEOUS ACOUSTIC SPEAKERS'] or 'unknown'} (capacity {CAP})",
              f"- Nemotron channels used: {ch}", f"- MISS {pc(mt.get('MISS'))} | FA {pc(mt.get('FA'))} | CONF {pc(mt.get('CONF'))} | DER {pc(mt.get('DER'))} (collar {mt.get('collar','n/a')}, overlap ignored: {mt.get('overlap_ignored','n/a')})",
              f"- Overlap: {run['analysis']['summary']['overlap_pct_of_speech']}% of speech"]
        for k in ("SPEAKER COLLAPSE", "SPEAKER ID INSTABILITY", "ASR", "PREPROCESSING", "STREAMING STATE", "SPEAKER CAPACITY", "POSTPROCESSING"):
            L.append(f"- {k}: **{F[k]['status']}** - " + "; ".join(F[k]["evidence"])[:400])
        L += ["- Suspicious timestamps: " + (", ".join(core.fmt(t) for t in sus) or "none"), ""]
        rows.append(f"| {m['input']} | {fu.get('marked','n/a')} | {fu.get('audible','n/a')} | {f['MAX SIMULTANEOUS ACOUSTIC SPEAKERS'] or '?'} | {ch} | {pc(mt.get('MISS'))} | {pc(mt.get('FA'))} | {pc(mt.get('CONF'))} | {pc(mt.get('DER'))} | {run['analysis']['summary']['overlap_pct_of_speech']}% | {F['SPEAKER COLLAPSE']['status']} | {F['SPEAKER ID INSTABILITY']['status']} | {m['tag']} |")
    L += ["## Aggregate", f"{ch_total} Nemotron speaker channels were observed across the evaluated recordings; {mk_total} people were marked (sum of per-recording entries, may double-count people). These are not expected to be equal.", "",
          "| Recording | Marked | Audible | Max simultaneous | Nemotron channels | MISS | FA | CONF | DER | Overlap | Collapse | ID instability | Notes |", "|" + "---|" * 13] + rows
    out = pathlib.Path(root) / "PSYCON_NEMOTRON_DIAGNOSTIC_REPORT.md"; out.write_text("\n".join(L)); return out
