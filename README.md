# Goldstein Studio – RunPod serverless workers for Real Cover (Pipeline A)

Until both endpoint IDs are filled in, Real Cover runs on the on-demand GPU pods (same RunPod API key).
As soon as **DEMUCS_ENDPOINT_ID** and **RVC_ENDPOINT_ID** are saved in
*AI Cover Studio → Real Cover → Serverless endpoints*, new covers use the endpoints automatically.

```
runpod_serverless/
  common.py            download/upload helpers (token-authenticated file exchange with your studio)
  demucs/Dockerfile    htdemucs_ft 4-stem separation (weights baked in)
  demucs/handler.py
  rvc/Dockerfile       Applio 3.6.6 RVC v2 inference (contentvec + rmvpe baked in)
  rvc/handler.py
```

## 1. Put the folder in a GitHub repository
1. Create a new (private is fine) GitHub repository, e.g. `grs-runpod-workers`.
2. Copy the contents of this `runpod_serverless` folder to the **root** of the repository
   (so `common.py`, `demucs/` and `rvc/` are at the top level) and push.

## 2. Create the Demucs endpoint
1. RunPod console → **Serverless** → **New Endpoint** → **GitHub Repo** (connect your GitHub account the first time).
2. Choose the repository and branch `main`.
3. Build settings: **Dockerfile path** `demucs/Dockerfile`, **Build context** `/` (repository root).
4. GPU: 24 GB (e.g. RTX 4090 / L4 / A5000). Max workers 1–2, Idle timeout 5 s, Execution timeout 1800 s.
5. Create the endpoint and wait until the build is **Ready**. Copy the **Endpoint ID** (the short id in the endpoint URL, e.g. `abc123xyz`).

## 3. Create the RVC endpoint
Same steps with **Dockerfile path** `rvc/Dockerfile`. GPU 24 GB is plenty. Copy its **Endpoint ID**.

## 4. Connect them to the studio
AI Cover Studio → **Real Cover (Pipeline A)** → *Serverless endpoints*: paste both IDs → **Save**.
The mode badge switches from **GPU pods** to **Serverless**. The RunPod API key of your enabled RunPod provider is used.

## Contract (for custom images)
Demucs input: `{"audio_url", "upload_url", "token", "model": "htdemucs_ft"}` – must PUT
`vocals`, `drums`, `bass`, `other` (WAV/FLAC) to `{upload_url}/{name}?token=…`.
RVC input: `{"vocal_url", "model_url", "index_url"|null, "pitch", "index_rate", "protect", "f0_method", "upload_url", "token"}` –
must PUT `converted`. Download URLs need `?token=…` appended. Return `{"error": "..."}` on failure.
Requests must send a non-default User-Agent (Cloudflare blocks `Python-urllib`).
