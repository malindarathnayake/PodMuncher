# syntax=docker/dockerfile:1
# Podmuncher offline ASR worker images (local development builds; no registry digest is claimed).
#
# Build from the repository root (the build context is limited by
# containers/asr.Dockerfile.dockerignore):
#   docker build -f containers/asr.Dockerfile --target cpu  -t podmuncher-asr:cpu-dev-local .
#   docker build -f containers/asr.Dockerfile --target cuda -t podmuncher-asr:cuda-dev-local .
#
# Architectures: the cpu target builds for linux/amd64 and linux/arm64 (Apple Silicon Docker
# Desktop). The pinned base is a multi-arch index, and each compiled wheel pin carries one
# --hash per architecture (manylinux x86_64, then manylinux aarch64). The cuda target is
# linux/amd64 only: the aarch64 ctranslate2 wheel has no CUDA, the NVIDIA pins below carry
# only x86_64 hashes, and Docker Desktop on a Mac has no GPU.
#
# The host runs these images read-only as 65532:65532 with --network none and the entrypoint
# ('python', '-I', '-m', 'podmuncher.asr_worker'); -I ignores PYTHONPATH, so the podmuncher
# modules are installed into site-packages below. No secrets, no ARG tokens, and no model
# weights are baked in: the verified CTranslate2 snapshot is mounted at /model at run time.
FROM python@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704 AS base

# Debian security fixes newer than the base image (python:3.12.15-slim-trixie).
RUN apt-get update \
 && apt-get upgrade -y --no-install-recommends \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Hash-pinned Python dependencies; --no-deps because every needed package is listed.
# PyAV (av) bundles its codecs, so no apt packages and no ffmpeg binary are needed.
COPY <<'REQUIREMENTS' /opt/asr-requirements.txt
faster-whisper==1.2.1 --hash=sha256:79a66ad50688c0b794dd501dc340a736992a6342f7f95e5811be60b5224a26a7
ctranslate2==4.8.2 --hash=sha256:554a0b2abf098a11c66ef7930c0a05b3de683a455e8ea3f2e6ab6966aaabc136 --hash=sha256:781f835da7fdc4adcc38f9d674dbe9b37eaf3c9aee3b2bc17d684342ff59d549
onnxruntime==1.31.0 --hash=sha256:5e129d6c56abd53e659cb70f00a108d6824086470ff99c2e47a82e5786563db3 --hash=sha256:35758d7606d578ec5b9d65f6e8a1f488013194c3f6097038a3223cb26d35ef9a
PyYAML==6.0.3 --hash=sha256:ba1cc08a7ccde2d2ec775841541641e4548226580ab850948cbfda66a1befcdc --hash=sha256:9149cad251584d5fb4981be1ecde53a1ca46c891a79788c0df828d2f166bda28
tokenizers==0.23.3 --hash=sha256:376851d22bcf9d650a5c3090bb83e6cf9e895fbf0595369fa4cd43c1f69b5f87 --hash=sha256:ec82e80e65a862275b97c3d90b7a523df8d9519ee48aeb4e9625b2cc909274e0
huggingface_hub==2.2.0 --hash=sha256:1667f145dc56dc210d60966069397df9ecfca9607a5d43db88b308c89dae56b3
click==8.5.0 --hash=sha256:255bc9599cf7748b4b1a446ccc735421bd08a2ae529a8b88597d3de5664ee360
hf-xet==1.7.0 --hash=sha256:2814a6e999d13464c4d679b788cc5d784eb5a4edfc638a31f10e9a11ab531ef8 --hash=sha256:fcfd6c22418e57dd5b3aea649e813b2e2cfb2aebf317b210d90f1fe4b3018b52
httpx2==2.13.1 --hash=sha256:6dff50fabc270ee5fd25d845d0b078ed20564579744d6d962850975996d2f9a4
httpcore2==2.13.1 --hash=sha256:e1e05d4f25f7d7d496bfb96748f6f4b67657b03da069b3a68c36069f3db73d0a
anyio==4.15.1 --hash=sha256:6152fdbbf9a77fdec97731721bebf7c4c44f7c29b424b0065826173efc7ed101
av==19.0.1 --hash=sha256:1bea5b6134209305199bce7627ac3d33964de2cf2b09c77d08e7f67cf8bd4170 --hash=sha256:437d4c0d5a7d771f2c3af84cd28e6aac6e173851116c60b53e81dbf1eebe4eab
filelock==4.2.1 --hash=sha256:7be2662e9e826b9bba28afc0f843a123280d799a2ea44bb5d35ccec67905ae44
fsspec==2026.9.0 --hash=sha256:8dd6e646e99ea382bd85f97a45e6b526a442d79423a7dc673f1e2756d05fcb5f
h11==0.16.0 --hash=sha256:63cf8bbe7522de3bf65932fda1d9c2772064ffb3dae62d55932da54b31cb6c86
idna==3.20 --hash=sha256:ab7ae7122974553370f0bdb919e1a960b2cd1bc1ef0276416d896db81c14582c
numpy==2.5.4 --hash=sha256:fbde6962867ee75b48b0ee29b2b9372ec5d617799dbaf38e82dc0596f2f7738a --hash=sha256:fe4d21ab149f15e4e6043dfb0de87e6e5f34ac176cde83060e9802981fca2ac2
packaging==26.3 --hash=sha256:d7193f7c8e4e93f444fde0262bf90af30e16fa0ad0ad44cb553c87339b23cd1c
protobuf==7.36.2 --hash=sha256:89f23aa53c24553a2416fd4fd1ec06f74fa42b14b546d8883128813f775bbfd2 --hash=sha256:e11e1f0180583a2af89db6a2ecd9e8dc40aa6d2988ca175bfd0e6d12ea72d74e
tqdm==4.70.1 --hash=sha256:c293e525e6fef9c20e8728fd4612df02a0aa31bb5fe91ecd93e123b1b7bffa73
truststore==0.10.5 --hash=sha256:9aaaedaefaf06d8b206278cf8b5012bc897f485a874503501e12d776df78951c
typing_extensions==4.16.0 --hash=sha256:481caa481374e813c1b176ada14e97f1f67a4539ce9cfeb3f350d78d6370c2e8
colorama==0.4.6 --hash=sha256:4f1d9991f5acc0ca119f9d443620b77f9d6b33703e51011c16baf57afb285fc6
flatbuffers==25.12.19 --hash=sha256:7634f50c427838bb021c2d66a3d1168e9d199b0607e6329399f04846d42e20b4
platformdirs==4.12.4 --hash=sha256:78bfb9db2a8471ed7eebe3c3c932da413911042994e699b384fbb4493fa872d7
REQUIREMENTS
# An optional build-time CA (BuildKit secret build_ca) is trusted for this one command only;
# it is never copied into a layer or trusted at runtime. The sha256 pins stay the integrity control.
RUN --mount=type=secret,id=build_ca,required=false if [ -f /run/secrets/build_ca ]; then export PIP_CERT=/run/secrets/build_ca; fi && pip install --no-cache-dir --no-deps --require-hashes -r /opt/asr-requirements.txt && rm /opt/asr-requirements.txt

# Only the ASR worker's import closure (not the full project, which pulls textual/otel):
# asr_worker -> protocol -> config (platformdirs) / requests / domains. No media_worker,
# yt-dlp or direct_download.
COPY src/podmuncher/__init__.py \
     src/podmuncher/domains.py \
     src/podmuncher/config.py \
     src/podmuncher/requests.py \
     src/podmuncher/protocol.py \
     src/podmuncher/asr_worker.py \
     /usr/local/lib/python3.12/site-packages/podmuncher/

# The Silero VAD model (vad_filter=True) ships inside the faster_whisper package: prove it is there.
RUN python -I -c "import podmuncher.asr_worker, faster_whisper, ctranslate2, av" \
 && python -I -c "import pathlib, faster_whisper; assets = pathlib.Path(faster_whisper.__file__).parent / 'assets'; found = sorted(assets.glob('silero_vad*.onnx')); assert found, 'VAD asset missing'; print(found[0].name)"

FROM base AS cpu
LABEL org.opencontainers.image.title="podmuncher-asr-cpu" \
      podmuncher.dev="1"
USER 65532:65532
ENTRYPOINT ["python", "-I", "-m", "podmuncher.asr_worker"]

FROM base AS cuda
# linux/amd64 only (see the header); do not build this target for linux/arm64.
# NVIDIA runtime libraries for CTranslate2 (CUDA 12 / cuDNN 9) from hashed wheels; the GPU
# itself and the driver are provided by the host (--gpus device=<uuid>).
COPY <<'REQUIREMENTS' /opt/asr-cuda-requirements.txt
nvidia-cublas-cu12==12.9.2.10 --hash=sha256:e4f53a8ca8c5d6e8c492d0d0a3d565ecb59a751b19cfdaa4f6da0ab2104c1702
nvidia-cudnn-cu12==9.27.0.42 --hash=sha256:0a4aa3a7d2264256506c6857fc41fc0c499982f78d70195b1cfcc9055c1957cd
nvidia-cuda-nvrtc-cu12==12.9.86 --hash=sha256:210cf05005a447e29214e9ce50851e83fc5f4358df8b453155d5e1918094dcb4
REQUIREMENTS
RUN --mount=type=secret,id=build_ca,required=false if [ -f /run/secrets/build_ca ]; then export PIP_CERT=/run/secrets/build_ca; fi && pip install --no-cache-dir --no-deps --require-hashes -r /opt/asr-cuda-requirements.txt && rm /opt/asr-cuda-requirements.txt
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib
LABEL org.opencontainers.image.title="podmuncher-asr-cuda" \
      podmuncher.dev="1"
USER 65532:65532
ENTRYPOINT ["python", "-I", "-m", "podmuncher.asr_worker"]
