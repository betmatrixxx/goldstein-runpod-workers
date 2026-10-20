FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update -qq && apt-get install -y -qq ffmpeg git build-essential && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch 3.6.6 https://github.com/IAHispano/Applio.git /app/applio
WORKDIR /app/applio
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir runpod==1.7.9 soundfile
# torchaudio>=2.9 saves through TorchCodec; install the matching build
RUN TV=$(python -c "import torch;print('.'.join(torch.__version__.split('+')[0].split('.')[:2]))") && \
    case "$TV" in 2.9) TC=0.9.1;; 2.10) TC=0.10.0;; 2.11) TC=0.11.1;; *) TC="";; esac && \
    if [ -n "$TC" ]; then pip install --no-cache-dir --no-deps torchcodec==$TC; fi
# bake the pretrained helpers (contentvec embedder, rmvpe) into the image
RUN python core.py prerequisites --models --no-exe
COPY common.py /app/common.py
COPY rvc/handler.py /app/handler.py
CMD ["python", "-u", "/app/handler.py"]
