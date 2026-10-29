<p align="center">
  <img alt="Podmuncher" src="Docs/Podmuncher%20branding%20kit/assets/png/podmuncher-lockup-horizontal-grey-1120x256.png" width="480">
</p>

<p align="center"><strong>Audio in. Text out.</strong><br>Open-source transcription that runs on your machine.</p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-F25F1C"></a>
  <img alt="Python 3.12 to 3.14" src="https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-1C1A17">
  <img alt="Windows and macOS" src="https://img.shields.io/badge/platform-Windows%20%7C%20macOS-6B655B">
</p>

Podmuncher turns podcasts, local audio and video files, direct HTTPS media links and public
YouTube videos into readable transcripts using local speech-to-text. Media never leaves your
machine for transcription: downloads, probing and transcription run in Docker containers, and
the transcription container has no network access.

It ships as a Python package with a command line, a guided setup and a terminal dashboard,
backed by a per-user background service that runs jobs one at a time.

## Features

- Transcribes local files, direct `https://` media URLs and individual public YouTube videos.
- Picks a speech-to-text model that fits your CPU, RAM and NVIDIA GPU, or uses the model you name.
- Runs on CUDA when an NVIDIA GPU is available, otherwise on the CPU.
- Writes plain, timestamped and readable text, SRT and JSON transcripts, and can keep the
  original video.
- Checks free storage before every stage and every published file.
- Verifies every output by size and hash, never overwrites existing files, and lists each output
  with its provenance in a per-job manifest.
- Retrying an export reuses the saved transcript instead of transcribing again.

## Requirements

- Windows or macOS (Apple Silicon). Linux hosts are not yet validated.
- Python 3.12, 3.13 or 3.14.
- Docker (Docker Desktop on Windows and macOS), installed and running. Setup never installs it.
- For GPU transcription: an NVIDIA GPU with a driver that supports Docker GPU passthrough.
  On macOS transcription always runs on the CPU; see [Docs/macos.md](Docs/macos.md).

## Install

Install the wheel attached to a [GitHub release](../../releases):

```sh
python -m pip install podmuncher-<version>-py3-none-any.whl
```

Container images are published to the GitHub Container Registry by the release workflow and
pulled during setup. There is no PyPI package.

To work from source:

```sh
python -m venv .venv
.venv/Scripts/python -m pip install --require-hashes -r requirements.lock   # Windows
.venv/Scripts/python -m pip install --no-deps -e ".[dev]"
```

On Linux and macOS use `.venv/bin/python`.

## Usage

Run the guided setup once. It checks Docker, storage and hardware, shows its plan per volume
before changing anything, pulls the images and downloads only the model it needs:

```sh
podmuncher --setup
```

Then transcribe:

```sh
podmuncher episode.mp3
podmuncher https://example.com/episode.mp3 --output ./transcripts
podmuncher "https://www.youtube.com/watch?v=VIDEO_ID" --video
```

Running `podmuncher` with no arguments in a terminal opens the dashboard. Other commands:

| Command | Purpose |
|---------|---------|
| `status` | Service status (never starts it) |
| `jobs`, `job ID` | List jobs or show one |
| `cancel ID`, `retry ID`, `clone ID INPUT` | Manage jobs |
| `recheck ID` | Refresh the preflight checks for a job |
| `doctor` | Diagnostics (never starts the service) |
| `service start\|stop` | Start or stop the background service |
| `cache list\|remove` | List or remove cache entries |
| `tui` | Attach the terminal dashboard |

Use `podmuncher COMMAND --help` for details.

## Development

```sh
.venv/Scripts/python -m pytest -q -m "not live"
.venv/Scripts/python -m ruff check src tests
```

Tests marked `live` need Docker, model files or a GPU and are skipped by the command above.

Before tagging a release, scan the locally built images and commit the reports in `security/`
(SBOMs and CVE scans):

```sh
.venv/Scripts/python bin/security_scan.py --platform linux-amd64
```

Release changes are listed in [CHANGELOG.md](CHANGELOG.md).

## License

Apache License 2.0. See [LICENSE](LICENSE). The branding artwork in
`Docs/Podmuncher branding kit/` is not covered by the code license; see
`Docs/Podmuncher branding kit/assets/README.md`.
