# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.2] - 2026-10-29

### Changed
- No setup is needed: `podmuncher <file or link>` uses the NVIDIA GPU when there is one (otherwise the CPU), picks a model that fits and downloads what it needs on first use.
- `podmuncher --setup` asks no questions. It applies the best settings for the machine; `--output`, `--device` and `--model` adjust them, and `--guided` brings back the step-by-step setup.
- Transcripts go to `~/Podmuncher` by default.
- A finished transcription prints where the transcript is, with plain progress steps instead of internal job fields; `--json` results include `output_dir` and `transcript`.

### Fixed
- GPU transcription works: the transcription step now runs on the selected GPU, and Docker Desktop's GPU passthrough (which hides the GPU's UUID from the container) is accepted.
- Jobs no longer fail at random with `storage_unknown`, `storage_stale` or `envelope_unvalidated` on Docker Desktop: storage and hardware readings may be up to 10 s old instead of 3 s, and two checks reading the same folder at once no longer lose its write permission.

## [1.0.1] - 2026-10-29

### Added
- The CUDA transcription image is published with every release, so setup can use an NVIDIA GPU.

### Fixed
- Setup now says when the release does not include the image for the chosen option, instead of asking for a development manifest.

## [1.0.0] - 2026-10-28

Podmuncher now works end to end on Windows and on macOS with Apple Silicon: a local service, command line, guided setup and terminal dashboard download and transcribe local files, direct HTTPS links and YouTube videos inside Docker containers. CUDA transcription has been measured on an NVIDIA GPU. Release images are built for linux/amd64 and linux/arm64 and are published only by the GitHub release workflow; until the first published release, the packaged release manifest lists what is still missing.

### Added
- Python package with lightweight help and version commands.
- Configuration validation and immutable request definitions for local files, HTTPS URLs, and individual YouTube videos.
- Resource and status snapshots that report values the system cannot determine as unknown or stale rather than as zero.
- Volume, quota, and Docker backing-storage checks before allocation, with conservative shared-storage budgets, overlap handling, and default reserves.
- Live CPU, RAM, and GPU measurements, including effective container limits, inform model selection.
- Workers can process media inside Docker containers, including jobs that need long timeouts approved for them.
- YouTube single videos and direct URLs can have their metadata read and their media downloaded and inspected inside a container, with duration limits enforced and merged video and audio that the size cap cut short rejected.
- Speech-to-text transcription runs fully offline on CPU from a verified local model, returning transcript segments, the detected language, and provenance.
- Models can be downloaded from Hugging Face and are saved only after the complete file is verified by size and hash, so partial files never appear as usable.
- Image builds can use a corporate TLS certificate as a build secret without baking it into the image, and build contexts include only the files needed.
- Job history, attempts, stages and outputs are kept in a local database on this machine, with network and shared folders refused and the database capped at 128 MiB.
- Finished job history is kept for 90 days or 10,000 jobs, whichever limit is reached first, and cleanup never deletes output files.
- Resubmitting a request with the same key returns the original job, and retrying it returns the same attempt instead of duplicate work.
- Jobs run one at a time in submission order, with the original video published before transcription and speech-to-text starting only once the audio is confirmed.
- Available storage is rechecked before every stage and before every published file.
- If a model runs out of memory in automatic mode, the job retries once with a smaller model.
- Export-only retries reuse the saved transcript and never transcribe the media again.
- Output files are checked for size, hash and content before publishing, then placed in a per-job folder that never overwrites an existing file and is listed in a manifest recording where each file came from.
- Stopping the service waits for workers to stop, marks the running job as interrupted, and keeps the queue for later.
- After a crash, the service removes its leftover containers before marking interrupted jobs, and nothing is released while a worker might still be running.
- A local background service accepts jobs over a loopback-only HTTP interface, survives the terminal closing, and can be started, checked and stopped from the command line.
- Command-line commands submit, follow, retry, recheck, clone and cancel jobs, list and remove cached models, and report status with documented exit codes and optional JSON output.
- Closing the terminal or pressing Ctrl+C while following a job never cancels it.
- `podmuncher doctor` runs read-only checks of paths, disk space, Docker, storage and the model cache without starting the service.
- Guided setup walks through locations, host and Docker checks, a size plan, explicit approval, model selection and a test run, and saves settings only after every step succeeds; it can also run unattended from an answers file.
- A terminal dashboard shows jobs, resources, models and setup with keyboard navigation, adapts to 80- and 120-column terminals, and asks for confirmation naming the job or cache entry before cancelling or deleting.
- Resource readings show their source and age, and the dashboard shows Unknown or Stale rather than zero when a value is missing or old.
- The service, command line, setup and dashboard are connected into one working app; `podmuncher setup` runs its downloads and checks through the local service, interactively or from an answers file.
- Automatic model selection picks the largest model that fits the machine from a shipped table of published and measured memory requirements, using the GPU when one is visible; any pinned model can be chosen explicitly, and setup lists the options with their sources.
- Eight Whisper models are pinned by exact revision and file hash: tiny, base, small, medium, distil-large-v3, large-v3-turbo, large-v2 and large-v3.
- macOS on Apple Silicon is supported with Docker Desktop: disk space is read from Docker Desktop's disk image, setup warns about folders outside Docker Desktop's file sharing, and transcription runs on the CPU image built for arm64.
- Media and CPU transcription images are built for linux/amd64 and linux/arm64; the CUDA image is linux/amd64 only and is never tried on an arm64 Docker engine.
- Setup checks free space on each volume separately, keeps settings it does not ask about when run again, and downloads the model into the cache folder you choose.
- An installed release loads its packaged release manifest, pulls its images during setup, and verifies each image by its registry digest; jobs keep working offline once the images and model are present.
- A GitHub release workflow tests on Windows, Linux and macOS, builds and publishes the multi-architecture images to the GitHub Container Registry, scans them, records a software bill of materials, and attaches the wheel to a GitHub Release.
- An Apache-2.0 LICENSE and a provenance inventory of third-party packages, images and models are included.

### Fixed
- Prevent Docker storage checks from waiting until timeout after completion, while preserving prompt cancellation and cleanup.
- Removed models now free their disk space on macOS and Linux instead of leaving renamed copies behind.
- CUDA no longer requires the exact NVIDIA driver version the release was measured with.
- Audio longer than a model's measured duration no longer rules that model out; the published memory requirement is used instead.
- The service raises its open-file limit at start, so macOS's default of 256 no longer causes failures.

### Security
- Worker-message validation with size limits and separate private error details.
- Bounded writes and retries with cleanup limited to owned processes and writes; storage reservations remain held until all outstanding writes are confirmed finished.
- Workers run in containers with no network access for local processing, a read-only root filesystem and all Linux capabilities dropped; on Linux they run as the invoking user and refuse to run as root, and on Windows they run as a fixed non-root user.
- Images and models run only when their pinned digests and hashes match.
- Worker images have no critical CVEs: they use Python 3.12.15 with current Debian security updates, and the media image ships a static FFmpeg 9.0.2 instead of Debian's FFmpeg. SBOMs and CVE scan reports are in `security/`.
- Model and media downloads must use HTTPS, follow at most 5 redirects, and check every hop and the actual connected server against private, loopback, and reserved addresses.
- Media downloads have byte and time limits and bounded retries with backoff, and responses whose end cannot be verified are refused because truncation could not be detected.
- Failed or unknown model files are quarantined and recorded in the audit log.
- Output from media inspection is size-capped.
- Only one service instance can run at a time, and its connection file is readable only by the current user.
- Local logs, traces and the audit log are size-capped and rotated, and tokens, passwords, file paths, web addresses and transcript text are removed from them.
- Exporting telemetry to an OpenTelemetry collector is off by default, retries at most three times, and never stores data for later delivery.
- If the audit log cannot be written, new work is blocked, but cancel and stop still work.
- The local service listens only on the loopback address, requires a per-start credential, refuses other hosts and browser origins, and limits request size, rate, connections and time.
- Text from media titles, errors and file names is escaped before display so terminal control sequences and direction-override characters cannot alter the screen.
