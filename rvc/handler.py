"""RunPod serverless handler: RVC v2 vocal conversion with Applio 3.6.6.

input:  {"vocal_url", "model_url", "index_url" (optional/null), "pitch", "index_rate", "protect", "f0_method", "upload_url", "token"}
output: {"converted_sec": ..., "seconds": ...} or {"error": "..."}
"""
import os
import sys
import tempfile
import time
import traceback

import runpod
import soundfile as sf

from common import download, upload

sys.path.insert(0, "/app/applio")
os.chdir("/app/applio")


def handler(job):
    inp = job.get("input") or {}
    t0 = time.time()
    try:
        for k in ("vocal_url", "model_url", "upload_url", "token"):
            if not inp.get(k):
                return {"error": f"missing input field: {k}"}
        work = tempfile.mkdtemp()
        vocal = download(inp["vocal_url"], inp["token"], os.path.join(work, "vocals.wav"))
        pth = download(inp["model_url"], inp["token"], os.path.join(work, "model.pth"))
        index = download(inp["index_url"], inp["token"], os.path.join(work, "model.index")) if inp.get("index_url") else ""
        out = os.path.join(work, "converted.wav")
        from core import run_infer_script
        msg = run_infer_script(pitch=int(inp.get("pitch", 0)), index_rate=float(inp.get("index_rate", 0.75)) if index else 0.0, volume_envelope=1,
                               protect=float(inp.get("protect", 0.33)), f0_method=inp.get("f0_method") or "rmvpe", input_path=vocal, output_path=out,
                               pth_path=pth, index_path=index, split_audio=False, f0_autotune=False, f0_autotune_strength=1.0, proposed_pitch=False,
                               proposed_pitch_threshold=155.0, clean_audio=False, clean_strength=0.7, export_format="WAV", embedder_model="contentvec",
                               embedder_model_custom=None, sid=0)
        if isinstance(msg, str) and "failed" in msg.lower() or not os.path.isfile(out):
            return {"error": f"Applio conversion failed: {msg}"}
        src_sec, sec = sf.info(vocal).duration, sf.info(out).duration
        if abs(sec - src_sec) > max(0.5, src_sec * 0.02):
            return {"error": f"converted vocal is {sec:.1f}s but input is {src_sec:.1f}s"}
        upload(inp["upload_url"], "converted", inp["token"], out)
        return {"converted_sec": round(sec, 2), "seconds": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]}


runpod.serverless.start({"handler": handler})
