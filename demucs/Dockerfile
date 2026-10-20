FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 TORCH_HOME=/models
RUN apt-get update -qq && apt-get install -y -qq ffmpeg && rm -rf /var/lib/apt/lists/*
# keep the image's CUDA build of torch/torchaudio (sm_50 … sm_120); never let pip replace it
RUN pip install --no-cache-dir runpod==1.7.9 soundfile==0.12.1 dora-search einops julius lameenc openunmix pyyaml tqdm && \
    pip install --no-cache-dir --no-deps demucs==4.0.1
# bake the htdemucs_ft weights into the image so cold starts don't download them
RUN python -c "from demucs.pretrained import get_model; get_model('htdemucs_ft')"
WORKDIR /app
COPY common.py /app/common.py
COPY demucs/handler.py /app/handler.py
CMD ["python", "-u", "/app/handler.py"]
