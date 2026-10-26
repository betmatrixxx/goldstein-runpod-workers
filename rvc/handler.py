"""RunPod serverless handler: RVC v2 vocal conversion with Applio 3.6.6.

input:  {"vocal_url", "model_url", "index_url" (optional/null), "pitch", "index_rate", "protect", "f0_method", "upload_url", "token"}
output: {"converted_sec": ..., "seconds": ...} or {"error": "..."}
"""
import json
import os
import sys
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

WORKER_VERSION = "2026-10-26"


def build_info():
    try:
        with open("/app/build_info.json") as f:
            b = json.load(f)
    except Exception:  # noqa: BLE001
        b = {}
    return {**b, "worker_version": WORKER_VERSION}


sys.path.insert(0, "/app/applio")
os.chdir("/app/applio")

# RVC's @torch.jit.script ops (fused_add_tanh_sigmoid_multiply) get JIT-compiled through NVRTC on the GPU;
# turn the GPU fusers off so inference never needs libnvrtc-builtins (plain CUDA kernels are used instead).
os.environ.setdefault("PYTORCH_JIT_USE_NNC_NOT_NVFUSER", "1")
try:
    import torch
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    if hasattr(torch._C, "_jit_set_nvfuser_enabled"):
        torch._C._jit_set_nvfuser_enabled(False)
    print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) \u00b7 JIT GPU fusion disabled", flush=True)
except Exception as e:  # noqa: BLE001
    print("could not disable JIT fusion:", e, flush=True)


def handler(job):
    inp = job.get("input") or {}
    t0 = time.time()
    try:
        for k in ("vocal_url", "model_url", "upload_url", "token"):
            if not inp.get(k):
                return {"error": f"missing input field: {k}"}
        work = tempfile.mkdtemp()
        info = gpu_report()
        info["build"] = build_info()
        if not info.get("cuda_ok"):
            return {"error": "CUDA is not usable on this worker (" + str(info.get("gpu")) + " " + str(info.get("capability")) + ", torch " + str(info.get("torch")) + " built for " + ",".join(info.get("arch_list") or []) + ")", "gpu": info}
        vocal = download(inp["vocal_url"], inp["token"], os.path.join(work, "vocals.wav"))
        pth = download(inp["model_url"], inp["token"], os.path.join(work, "model.pth"))
        index = download(inp["index_url"], inp["token"], os.path.join(work, "model.index")) if inp.get("index_url") else ""
        out = os.path.join(work, "converted.wav")
        style = (inp.get("style_tag") or "")[:300]
        print(f"style_tag={style!r}", flush=True)
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
        return {"converted_sec": round(sec, 2), "style_tag": style, "gpu": info, "seconds": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]}


runpod.serverless.start({"handler": handler})
