FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update -qq && apt-get install -y -qq ffmpeg git build-essential && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch 3.6.6 https://github.com/IAHispano/Applio.git /app/applio
WORKDIR /app/applio
# keep the image's torch 2.7.1+cu128 (its NVRTC matches); Applio's requirements must not swap in a CUDA 13 torch
RUN python -c "import torch,torchaudio; print(f'torch=={torch.__version__}\ntorchaudio=={torchaudio.__version__}')" > /tmp/torch-pin.txt && \
    (python -c "import torchvision; print(f'torchvision=={torchvision.__version__}')" >> /tmp/torch-pin.txt || true) && cat /tmp/torch-pin.txt && \
    grep -viE "^(torch|torchaudio|torchvision)([=<>~! ;\[]|$)" requirements.txt > /tmp/req.txt && \
    pip install --no-cache-dir -r /tmp/req.txt -c /tmp/torch-pin.txt && pip install --no-cache-dir -c /tmp/torch-pin.txt runpod==1.7.9 soundfile && \
    python -c "import torch,sys; print('torch after Applio deps:', torch.__version__, torch.version.cuda); sys.exit(0 if str(torch.version.cuda).startswith('12') else 1)"
# torchaudio>=2.9 saves through TorchCodec; install the matching build
RUN TV=$(python -c "import torch;print('.'.join(torch.__version__.split('+')[0].split('.')[:2]))") && \
    case "$TV" in 2.9) TC=0.9.1;; 2.10) TC=0.10.0;; 2.11) TC=0.11.1;; *) TC="";; esac && \
    if [ -n "$TC" ]; then pip install --no-cache-dir --no-deps torchcodec==$TC; fi
# bake the pretrained helpers (contentvec embedder, rmvpe) into the image
RUN python core.py prerequisites --models --no-exe
COPY common.py /app/common.py
COPY rvc/handler.py /app/handler.py
RUN python -c "import json,datetime,torch; json.dump({'build_date': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'), 'torch': torch.__version__, 'cuda_build': torch.version.cuda}, open('/app/build_info.json','w'))" && cat /app/build_info.json
CMD ["python", "-u", "/app/handler.py"]
