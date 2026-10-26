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

try:
    from common import gpu_report
except ImportError:  # older common.py in the repo — report the GPU locally instead of crashing
    def gpu_report():
        r = {}
        try:
            import torch
            r.update(torch=torch.__version__, cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available())
            if r["cuda_available"]:
                r.update(gpu=torch.cuda.get_device_name(0), capability="sm_%d%d" % torch.cuda.get_device_capability(0))
                r["cuda_ok"] = float((torch.arange(4, device="cuda") * 2).sum()) == 12.0
        except Exception as e:  # noqa: BLE001
            r.update(cuda_ok=False, error=f"{type(e).__name__}: {e}"[:300])
        print("GPU report:", r, flush=True)
        return r

STEMS = ("vocals", "drums", "bass", "other")


def demucs(model, device, wav, out):
    return subprocess.run(["python", "-m", "demucs", "-n", model, "-d", device, "-o", out, wav], capture_output=True, text=True)


def short(p):
    lines = (p.stderr or p.stdout or "").splitlines()
    return "\n".join([x for x in lines if "Warning" in x or "capability" in x][:4] + lines[-12:])[-1800:]


def handler(job):
    inp = job.get("input") or {}
    t0 = time.time()
    try:
        for k in ("audio_url", "upload_url", "token"):
            if not inp.get(k):
                return {"error": f"missing input field: {k}"}
        model = inp.get("model") or "htdemucs_ft"
        info = gpu_report()
        work = tempfile.mkdtemp()
        src = download(inp["audio_url"], inp["token"], os.path.join(work, "input.audio"))
        wav = os.path.join(work, "input.wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-ac", "2", "-ar", "44100", wav], check=True)
        device = "cuda" if info.get("cuda_ok") else "cpu"
        p = demucs(model, device, wav, os.path.join(work, "sep"))
        if p.returncode and device == "cuda" and "CUDA" in (p.stderr or ""):
            info["cuda_error"] = short(p)[-600:]
            device = "cpu"
            p = demucs(model, device, wav, os.path.join(work, "sep"))
        if p.returncode:
            return {"error": f"demucs failed on {device}: " + short(p), "gpu": info}
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
        return {"stems": out, "model": model, "device": device, "gpu": info, "seconds": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]}


runpod.serverless.start({"handler": handler})
