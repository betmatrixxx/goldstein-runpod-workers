"""Shared helpers for the Goldstein Studio RunPod serverless workers."""
import os
import urllib.request

UA = {"User-Agent": "GoldsteinStudio-GPUWorker/1.0"}


def download(url, token, dst):
    sep = "&" if "?" in url else "?"
    req = urllib.request.Request(f"{url}{sep}token={token}", headers=UA)
    with urllib.request.urlopen(req, timeout=600) as r, open(dst, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    if os.path.getsize(dst) == 0:
        raise RuntimeError(f"downloaded file is empty: {url.split('?')[0]}")
    return dst


def upload(upload_url, name, token, path):
    if not os.path.isfile(path) or os.path.getsize(path) < 1024:
        raise RuntimeError(f"{name}: output file missing or empty ({path})")
    with open(path, "rb") as f:
        data = f.read()
    req = urllib.request.Request(f"{upload_url}/{name}?token={token}", data=data, method="PUT",
                                 headers={**UA, "Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return r.read().decode()
