import json, pathlib, tempfile
import pandas as pd, streamlit as st, streamlit.components.v1 as components
import core, diagnostics

st.set_page_config(page_title="Nemotron Diarization Bench", layout="wide")
st.title("Nemotron-3-Diarization benchmark")

with st.sidebar:
    preset = st.selectbox("Mode", list(core.PRESETS))
    cfg = {k: st.number_input(k + " (80 ms frames)", value=v, key=preset + k) for k, v in core.PRESETS[preset].items()}
    thr = st.slider("Activity threshold", 0.1, 0.9, 0.5, 0.05)
    asr = st.checkbox("Run ASR (faster-whisper)", True); asr_size = st.text_input("ASR model", "large-v3")
    allow_cpu = st.checkbox("Allow CPU (same model, slow)", False)
    tag = st.selectbox("Test tag", ["", "TEST 1 (2spk clean)", "TEST 2 (4spk clean)", "TEST 3 (8spk clean)", "TEST 4 (4spk interruptions)",
                                    "TEST 5 (4spk overlap)", "TEST 6 (8spk overlap)", "TEST 7 (PSYCON)"])

st.header("UPLOAD RECORDING")
up = st.file_uploader("Upload Video/Audio", type=["mp4", "mov", "mkv", "avi", "webm", "wav", "mp3", "m4a", "flac", "ogg", "opus"])
if up and st.button("RUN NEMOTRON", type="primary"):
    tmp = pathlib.Path(tempfile.mkdtemp()) / up.name; tmp.write_bytes(up.getbuffer())
    try:
        with st.spinner("Preprocessing + diarizing + ASR..."):
            st.session_state.run = str(core.run_all(tmp, preset, {k: int(v) for k, v in cfg.items()}, thr, asr, asr_size, None, allow_cpu, tag))
    except Exception as e:
        st.error(f"Run failed: {e}")

runs = sorted(core.RUNS.glob("*/run.json"), reverse=True) if core.RUNS.exists() else []
if not runs:
    st.stop()
names = [r.parent.name for r in runs]
cur = st.session_state.get("run", str(runs[0].parent))
sel = st.selectbox("Run", names, index=names.index(pathlib.Path(cur).name) if pathlib.Path(cur).name in names else 0)
rd = core.RUNS / sel
run = json.load(open(rd / "run.json")); man, an, segs = run["manifest"], run["analysis"], run["segments"]
groups = json.load(open(rd / "transcript.json")); media = next(rd.glob("media.*"))
is_video = media.suffix in {".mp4", ".mov", ".mkv", ".avi", ".webm"}
url = f"app/static/runs/{sel}/" + (media.name if is_video else "audio_16k_mono.wav")

st.header("VIDEO PLAYER | SPEAKER TIMELINE")
dur = man["preprocessing"]["duration_s"]
html = f"""
<style>body{{font:13px sans-serif;margin:0}}#tl{{position:relative;height:{8*24+4}px;background:#f3f3f3;margin-top:6px}}
.b{{position:absolute;height:20px;opacity:.85;cursor:pointer;border-radius:2px}}.o{{position:absolute;top:0;bottom:0;background:rgba(255,0,0,.25)}}
.l{{position:absolute;left:2px;font-size:10px;color:#555}}</style>
<{'video' if is_video else 'audio'} id=m src="{url}" controls style="width:100%;max-height:300px"></{'video' if is_video else 'audio'}>
<div id=tl></div><pre id=info style="white-space:pre-wrap">Click a block.</pre>
<script>
const D={dur},S={json.dumps(segs)},O={json.dumps(an['overlap_timeline'])},G={json.dumps(groups)};
const tl=document.getElementById('tl'),m=document.getElementById('m');const SEEK={st.session_state.get('seek', 0)};m.addEventListener('loadedmetadata',()=>{{if(SEEK>0)m.currentTime=SEEK}});
O.forEach(o=>tl.insertAdjacentHTML('beforeend',`<div class=o style="left:${{o.start/D*100}}%;width:${{Math.max(o.duration/D*100,.15)}}%"></div>`));
for(let i=0;i<8;i++)tl.insertAdjacentHTML('beforeend',`<div class=l style="top:${{i*24+6}}px">Spk ${{i}}</div>`);
S.forEach(s=>{{const d=document.createElement('div');d.className='b';
d.style.cssText=`left:${{s.start/D*100}}%;width:${{Math.max(s.duration/D*100,.15)}}%;top:${{s.speaker*24+2}}px;background:hsl(${{s.speaker*45}},70%,45%)`;
d.onclick=()=>{{m.currentTime=s.start;m.play();const stop=()=>{{if(m.currentTime>=s.end){{m.pause();m.removeEventListener('timeupdate',stop)}}}};m.addEventListener('timeupdate',stop);
const t=G.filter(g=>g.end>s.start&&g.start<s.end).map(g=>(g.overlap?'[OVERLAPPING SPEECH - TRANSCRIPTION UNCERTAIN] ':'')+g.text).join(' | ');
document.getElementById('info').textContent=`Speaker ${{s.speaker}}  ${{s.start.toFixed(2)}}-${{s.end.toFixed(2)}}s  (mean prob ${{s.confidence.toFixed(2)}})\\n${{t||'(no ASR text)'}}`}};tl.appendChild(d)}});
</script>"""
components.html(html, height=460, scrolling=False)
st.caption("Red bands = overlap (>=2 channels active). Labels are anonymous, not people. MOV/odd codecs may not play in-browser; analysis is unaffected.")

st.header("SPEAKER-ATTRIBUTED TRANSCRIPT")
st.code((rd / "transcript.txt").read_text() or "(ASR disabled)", language=None)

st.header("DIAGNOSTIC METRICS")
s = an["summary"]; c = st.columns(4)
c[0].metric("Duration s", s["duration_s"]); c[1].metric("Speech s", s["speech_s"])
c[2].metric("Overlap s", s["overlap_s"]); c[3].metric("Overlap % of speech", s["overlap_pct_of_speech"])
st.metric("Speakers detected (>=1 s speech)", s["detected_speakers"])
st.dataframe(pd.DataFrame(an["speaker_stats"]), hide_index=True)
for f in an["flags"]: st.warning(f)
with st.expander("Overlap timeline"): st.dataframe(pd.DataFrame(an["overlap_timeline"]))
with st.expander("Run manifest (reproducibility)"): st.json(man)

st.subheader("Ground truth")
gt = st.file_uploader("Reference RTTM", type=["rttm"]); c1, c2 = st.columns(2)
collar = c1.number_input("Collar (s)", 0.0, 1.0, 0.0, 0.05); skip = c2.radio("Overlap", ["Included", "Ignored"]) == "Ignored"
if gt:
    (rd / "ref.rttm").write_bytes(gt.getbuffer())
if (rd / "ref.rttm").exists():
    m = core.der(rd / "ref.rttm", segs, dur, collar, skip)
    json.dump(m, open(rd / "metrics.json", "w"), indent=1)
    st.json({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
    st.caption("DER = MISS + FA + CONF as fractions of reference speech (pyannote.metrics). Overlap mode + collar are stated above.")

st.subheader("Manual speaker -> person mapping (kept separate from raw output)")
ids = sorted({g["speaker"] for g in segs}); mp = rd / "person_map.json"
old = json.load(open(mp)) if mp.exists() else {}
ed = st.data_editor(pd.DataFrame({"speaker": [f"Speaker {i}" for i in ids], "person": [old.get(f"Speaker {i}", "") for i in ids]}), hide_index=True)
if st.button("Save mapping"): json.dump(dict(zip(ed.speaker, ed.person)), open(mp, "w"))

def ts_buttons(times, key):
    if times:
        cols = st.columns(min(len(times), 8))
        for i, t in enumerate(times[:8]):
            if cols[i].button(core.fmt(t), key=f"{key}{i}"): st.session_state.seek = t; st.rerun()


st.header("DIAGNOSTICS: COLLAPSE / IDENTITY / CAPACITY")
st.caption("Everything here is derived from the untouched raw probabilities. Marked people are not the expected Nemotron speaker count.")
fp = rd / "funnel.json"; fu = json.load(open(fp)) if fp.exists() else {}
c = st.columns(5)
fu["marked"] = c[0].number_input("Marked people", 0, 1000, fu.get("marked", 0)); fu["present"] = c[1].number_input("Actually present", 0, 1000, fu.get("present", 0))
fu["audible"] = c[2].number_input("Est. audible", 0, 1000, fu.get("audible", 0)); fu["max_simul"] = c[3].number_input("Known max simultaneous (0=unknown)", 0, 1000, fu.get("max_simul", 0))
fu["usable_min_s"] = c[4].number_input("Usable profile = min speech s", 0, 3600, fu.get("usable_min_s", 10)); json.dump(fu, open(fp, "w"))
dg = diagnostics.diagnose(rd)

st.subheader("Failure attribution (no single verdict)")
for f in dg["failure_attribution"]:
    with st.expander(f"{f['name']}: {f['status']}", expanded=f["status"] not in ("PASS", "LOW", "NOT DETECTED", "NO EVIDENCE", "CONSISTENT")):
        for e in f["evidence"]: st.write("- " + e)
        ts_buttons(f["times"], f["name"])

st.subheader("Speaker-count funnel")
fn = dg["funnel"]; mx = max([v for _, v in fn if v] or [1])
st.markdown("".join(f"<div style='margin:3px 0'><div style='background:#3a7;color:#fff;padding:3px 8px;width:{max(8, 100*(v or 0)/mx)}%;white-space:nowrap'>{l}: {v if v is not None else 'not entered'}</div></div>" for l, v in fn), unsafe_allow_html=True)
ms = dict(fn)["MAX SIMULTANEOUS ACOUSTIC SPEAKERS"]
if ms: (st.warning if ms > diagnostics.CAP else st.info)(f"Maximum simultaneous speakers: {ms} | Nemotron capacity: {diagnostics.CAP}" + (" | WARNING: speaker capacity may affect interpretation" if ms > diagnostics.CAP else ""))

st.subheader("Simultaneous speakers")
sd = {"Nemotron": dg["simultaneous_nemotron"], **({"Ground truth": dg["gt_dist"]} if "gt_dist" in dg else {})}
for k, v in sd.items(): st.write(f"**{k}**: max {v['max']}, mean during speech {v['mean_during_speech']:.2f}, % of speech with k speakers: {v['pct_of_speech']}")

if "forward" in dg:
    st.subheader("Ground truth -> Nemotron mapping"); st.dataframe(pd.DataFrame(dg["forward"]), hide_index=True)
    st.dataframe(pd.DataFrame(dg["inverse"]), hide_index=True)
    for cl in dg["collapse"]:
        (st.error if cl["strength"] == "STRONG" else st.warning)(f"POSSIBLE SPEAKER COLLAPSE ({cl['strength']}): {' and '.join(cl['persons'])} share Speaker {cl['channel']}. Type(s): {'; '.join(cl['types'])}\n\n" + "\n".join("- " + e for e in cl["evidence"]))
st.subheader("Pairwise channel relationships"); st.dataframe(pd.DataFrame(dg["pairwise"]), hide_index=True)
st.subheader("Chunk-boundary identity diagnostics")
bd = pd.DataFrame(dg["boundaries"]); st.write(f"{len(bd)} boundaries; flagged shown first")
if len(bd): st.dataframe(pd.concat([bd[bd.status.str.startswith(("POSSIBLE", "CONFIRMED"))], bd[~bd.status.str.startswith(("POSSIBLE", "CONFIRMED"))]]).head(300), hide_index=True)
wp = rd / "words.json"
if wp.exists():
    with st.expander("ASR words: raw ASR timing vs Nemotron activity vs heuristic attribution"): st.dataframe(pd.DataFrame(json.load(open(wp))))
if st.button("Generate PSYCON_NEMOTRON_DIAGNOSTIC_REPORT.md"):
    rp = diagnostics.build_report(core.RUNS); st.download_button("Download report", rp.read_bytes(), rp.name)

st.header("Downloads")
d = st.columns(4)
d[0].download_button("DOWNLOAD JSON", (rd / "run.json").read_bytes(), "run.json")
d[1].download_button("DOWNLOAD RTTM", (rd / "hyp.rttm").read_bytes(), "hyp.rttm")
d[2].download_button("DOWNLOAD CSV", (rd / "segments.csv").read_bytes(), "segments.csv")
d[3].download_button("DOWNLOAD TRANSCRIPT", (rd / "transcript.txt").read_bytes(), "transcript.txt")

st.header("Comparison table")
rows = []
for r in core.RUNS.glob("*/run.json"):
    j = json.load(open(r)); mt = json.load(open(r.parent / "metrics.json")) if (r.parent / "metrics.json").exists() else {}
    rows.append({"Recording": j["manifest"]["input"], "Test": j["manifest"]["tag"], "Mode": j["manifest"]["preset"], "True speakers": mt.get("true_speakers"),
                 "Nemotron detected": j["analysis"]["summary"]["detected_speakers"], "MISS": mt.get("MISS"), "FA": mt.get("FA"), "CONF": mt.get("CONF"),
                 "DER": mt.get("DER"), "Overlap %": j["analysis"]["summary"]["overlap_pct_of_speech"], "Flags": len(j["analysis"]["flags"])})
st.dataframe(pd.DataFrame(rows), hide_index=True)
