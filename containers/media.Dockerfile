# syntax=docker/dockerfile:1
# Podmuncher media worker image (local development build; no registry digest is claimed).
#
# Build from the repository root:
#   docker build -f containers/media.Dockerfile -t podmuncher-media:dev-local .
#
# Multi-arch: linux/amd64 and linux/arm64 (Apple Silicon). The base digest (python:3.12.15-slim-trixie)
# and the static FFmpeg image are multi-arch indexes, and every pip pin carries one sha256 per
# architecture-specific wheel (pure-Python wheels have one). Add --platform to cross-build.
#
# The host runs this image read-only as 65532:65532 with the entrypoint
# ('python', '-I', '-m', 'podmuncher.media_worker'); -I ignores PYTHONPATH, so the
# podmuncher modules are installed into site-packages below. No secrets, no ARG
# tokens, no cookies, and nothing here is fetched at runtime.
FROM python@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704

LABEL org.opencontainers.image.title="podmuncher-media" \
      podmuncher.dev="1"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Debian security fixes newer than the base image.
RUN apt-get update \
 && apt-get upgrade -y --no-install-recommends \
 && rm -rf /var/lib/apt/lists/*

# Static FFmpeg and ffprobe (mwader/static-ffmpeg 9.0.2-1, multi-arch index pinned by digest).
# Debian's ffmpeg carries unfixed critical CVEs and pulls in libtiff/libxml2/X11 libraries.
COPY --from=mwader/static-ffmpeg@sha256:f5d100184884eb3e82b0eb14bb5ebf98da99826ac824989720297fb9ef21c822 \
     /ffmpeg /ffprobe /usr/local/bin/
RUN ffprobe -version

# Hash-pinned Python dependencies; --no-deps because every needed package is listed.
# deno 2.9.5 is the version yt-dlp's own pin-deno extra pins; yt-dlp-ejs supplies the
# local JS challenge solver so remote component fetching stays disabled.
# deno hashes: manylinux_2_27_x86_64 then manylinux_2_27_aarch64 wheel (PyPI digests).
COPY <<'REQUIREMENTS' /opt/media-requirements.txt
yt-dlp==2026.8.19 --hash=sha256:1d57897e94c6665a0a6f9bc54b34e584284e32c034ffab3a7df25d8f7b24eedf
yt-dlp-ejs==0.8.0 --hash=sha256:79300e5fca7f937a1eeede11f0456862c1b41107ce1d726871e0207424f4bdb4
deno==2.9.5 --hash=sha256:e5aec0419739fd0359663a78765d4feda19d1d0b60a682d25add9b78e19f95b4 --hash=sha256:bccec0965917b6daa323f9ce6ad8e884e0854506b0cdaec9de41bdac7620109a
platformdirs==4.12.4 --hash=sha256:78bfb9db2a8471ed7eebe3c3c932da413911042994e699b384fbb4493fa872d7
REQUIREMENTS
# An optional build-time CA (BuildKit secret build_ca) is trusted for this one command only;
# it is never copied into a layer or trusted at runtime. The sha256 pins stay the integrity control.
RUN --mount=type=secret,id=build_ca,required=false \
    if [ -f /run/secrets/build_ca ]; then export PIP_CERT=/run/secrets/build_ca; fi \
 && pip install --no-cache-dir --no-deps --require-hashes -r /opt/media-requirements.txt \
 && rm /opt/media-requirements.txt

# Only the media worker's import closure (not the full project, which pulls textual/otel):
# media_worker -> protocol -> config (platformdirs) / requests / domains; direct_download -> domains.
COPY src/podmuncher/__init__.py \
     src/podmuncher/domains.py \
     src/podmuncher/config.py \
     src/podmuncher/requests.py \
     src/podmuncher/protocol.py \
     src/podmuncher/direct_download.py \
     src/podmuncher/media_worker.py \
     /usr/local/lib/python3.12/site-packages/podmuncher/

RUN python -I -c "import podmuncher.media_worker, yt_dlp, yt_dlp_ejs" \
 && ffprobe -version \
 && python -I -c "import json, podmuncher.media_worker as m; print(json.dumps(m.verify_installed_yt_dlp(), sort_keys=True)); import subprocess; subprocess.run([m.deno_path(), '--version'], check=True)"

USER 65532:65532
ENTRYPOINT ["python", "-I", "-m", "podmuncher.media_worker"]
