"""RunPod serverless handler: Demucs 4-stem separation (htdemucs_ft).

input:  {"audio_url", "upload_url", "token", "model": "htdemucs_ft"}
output: {"stems": {"vocals": secs, ...}, "model": ..., "seconds": ...} or {"error": "..."}
"""
import os
import subprocess
import tempfile
import time
import traceback

import runpod
import soundfile as sf

from common import download, upload

STEMS = ("vocals", "drums", "bass", "other")


def handler(job):
    inp = job.get("input") or {}
    t0 = time.time()
    try:
        for k in ("audio_url", "upload_url", "token"):
            if not inp.get(k):
                return {"error": f"missing input field: {k}"}
        model = inp.get("model") or "htdemucs_ft"
        work = tempfile.mkdtemp()
        src = download(inp["audio_url"], inp["token"], os.path.join(work, "input.audio"))
        wav = os.path.join(work, "input.wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-ac", "2", "-ar", "44100", wav], check=True)
        p = subprocess.run(["python", "-m", "demucs", "-n", model, "-d", "cuda", "-o", os.path.join(work, "sep"), wav], capture_output=True, text=True)
        if p.returncode:
            return {"error": "demucs failed: " + (p.stderr or p.stdout)[-1500:]}
        base = os.path.join(work, "sep", model, "input")
        src_sec = sf.info(wav).duration
        out = {}
        for s in STEMS:
            path = os.path.join(base, s + ".wav")
            sec = sf.info(path).duration
            if abs(sec - src_sec) > max(0.5, src_sec * 0.02):
                return {"error": f"{s} stem is {sec:.1f}s but input is {src_sec:.1f}s"}
            upload(inp["upload_url"], s, inp["token"], path)
            out[s] = round(sec, 2)
        return {"stems": out, "model": model, "seconds": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]}


runpod.serverless.start({"handler": handler})
