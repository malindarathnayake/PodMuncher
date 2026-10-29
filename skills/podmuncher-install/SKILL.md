---
name: podmuncher-install
description: Install and set up Podmuncher (local podcast and video transcription in Docker) on this machine from the latest GitHub release. Use when the user asks to install, set up, reinstall or repair Podmuncher.
---

# Install Podmuncher

Podmuncher turns local audio/video files, direct `https://` media links and public YouTube
videos into transcripts with local speech-to-text. It is a Python command-line tool backed by a
per-user background service; all media work runs in Docker containers pulled from
`ghcr.io/malindarathnayake`.

Follow the steps in order. Run every command yourself and read its output before moving on.
Stop and ask the user whenever a step says so.

## Rules

- Do not install Docker, Python or GPU drivers yourself, and do not change Docker settings.
  If one is missing, tell the user exactly what to install and stop.
- Do not run anything as administrator or root. Podmuncher installs per user.
- Do not edit Podmuncher's config files by hand; setup writes them.
- When a Podmuncher command prints a `next_action`, relay it to the user verbatim.
- Shell: on Windows use PowerShell; on macOS and Linux use bash/zsh.

## 1. Check the prerequisites

| Check | Command | Needs |
|---|---|---|
| OS | — | Windows 10/11 or macOS on Apple Silicon. Linux is not validated yet: warn the user and ask before continuing. |
| Python | `py -3.12 --version` (Windows) or `python3 --version` | 3.12, 3.13 or 3.14 |
| Docker | `docker info --format "{{.ServerVersion}}"` | Prints a version. Docker Desktop must be installed and running. |
| Disk | — | About 3 GB free for CPU setup, about 10 GB for GPU setup, plus room for models and outputs. |
| NVIDIA GPU (optional) | `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader` | Windows only. A GPU listed here can be used for transcription. |

If Python is missing or older than 3.12, ask the user to install Python 3.12+ from python.org.
If `docker info` fails, ask the user to install or start Docker Desktop, then re-check.

## 2. Defaults (do not interrogate the user)

Podmuncher picks the best settings itself: the NVIDIA GPU when one is present (CPU as the
fallback), otherwise the CPU, and a model that fits the hardware. Transcripts go to
`~/Podmuncher`. Only ask if the user has already said they want something different, for
example a specific transcript folder or CPU only. Never use a network or cloud-synced
folder (OneDrive, iCloud, Dropbox) for output; Podmuncher refuses shared folders.

## 3. Install the package into its own virtual environment

Windows (PowerShell):

```powershell
py -3.12 -m venv "$HOME\podmuncher-venv"
$release = Invoke-RestMethod https://api.github.com/repos/malindarathnayake/PodMuncher/releases/latest
$wheel = ($release.assets | Where-Object name -like 'podmuncher-*.whl').browser_download_url
& "$HOME\podmuncher-venv\Scripts\python" -m pip install --upgrade $wheel
$pm = "$HOME\podmuncher-venv\Scripts\podmuncher.exe"
& $pm --version
```

macOS (bash/zsh):

```sh
python3 -m venv ~/podmuncher-venv
WHEEL=$(curl -fsSL https://api.github.com/repos/malindarathnayake/PodMuncher/releases/latest \
  | python3 -c 'import json,sys; print(next(a["browser_download_url"] for a in json.load(sys.stdin)["assets"] if a["name"].endswith(".whl")))')
~/podmuncher-venv/bin/python -m pip install --upgrade "$WHEEL"
PM=~/podmuncher-venv/bin/podmuncher
"$PM" --version
```

`--version` must print `podmuncher <version>`. Use the full path (`$pm` / `$PM`) for every
later command; do not rely on the venv being activated.

## 4. Run the read-only health check

```powershell
& $pm doctor --json
```

`doctor` changes nothing. It should return `"ok": true` and `"exit_code": 0`, and the
`paths`, `host_capacity`, `docker`, `backing` and `cache` entries under `checks` should have
`"status": "ok"`. Before setup, `service` is `stopped` and `snapshot` is `stale`; that is
normal. For any other check that is not ok, report its `name`, `reason_code` and the top-level
`next_action` to the user and stop until it is fixed. The common one is `docker`
(`docker_unavailable`): Docker Desktop is not running.

If a Podmuncher service from an earlier install is running (`podmuncher status --json` shows
`"service_state": "running"`), run `podmuncher service stop` first so setup starts the new
version.

## 5. Download what transcription needs

Tell the user this downloads about 0.5–2 GB (CPU) or 2–5 GB (GPU), depending on the model,
and starts the Podmuncher background service. Then run it; it asks no questions:

```powershell
& $pm --setup
```

Add `--output "D:\Transcripts"`, `--device cpu` or `--model small` only if the user asked
for them. It prints what it picked, then `Ready. Transcripts go to <folder>`. Exit 0 means
done. Exit 3 means blocked: relay the printed `setup: blocked (...) - <next action>` line to
the user. Exit 4 means it failed: run it once more, then report the line and
`doctor --json`.

This step is optional: the first transcription downloads whatever is missing on its own.
Running it now just makes the first transcription quicker. It can take several minutes.

## 6. Verify with a real transcription

Ask the user for a short media file or a YouTube link (a minute or two is enough) and
transcribe it:

```powershell
& $pm "C:\path\to\clip.mp3" --json
```

The JSON result has `ok`, `output_dir` and `transcript` (the path of the plain-text
transcript). Show the user where the transcript is and its first few lines.

## 7. Tell the user how to use it

- Transcribe: `podmuncher <file-or-url>`; add `--video` to keep the original video.
- Dashboard: run `podmuncher` with no arguments in a terminal.
- Status and jobs: `podmuncher status`, `podmuncher jobs`.
- Service: `podmuncher service start|stop`; `podmuncher status` shows whether it is running.
- Change settings: `podmuncher --setup --output DIR` (or `--device`, `--model`).
  `podmuncher --setup --guided` asks before each step.

The command lives at `~/podmuncher-venv/Scripts/podmuncher.exe` (Windows) or
`~/podmuncher-venv/bin/podmuncher` (macOS). Offer to add that folder to the user's PATH;
only do it if they agree.

## 8. Let the agent use Podmuncher later

Install the companion `podmuncher` usage skill so future sessions can transcribe on request.
Install it for the agent you are running in (both if unsure):

| Agent | Skill folder |
|---|---|
| Claude Code | `~/.claude/skills/podmuncher/` |
| Codex CLI | `~/.codex/skills/podmuncher/` |

Windows (PowerShell):

```powershell
$url = "https://github.com/malindarathnayake/PodMuncher/releases/latest/download/podmuncher-SKILL.md"
foreach ($agent in "$HOME\.claude", "$HOME\.codex") {
  if (-not (Test-Path $agent)) { continue }   # that agent is not installed
  $dir = "$agent\skills\podmuncher"
  New-Item -ItemType Directory -Force $dir | Out-Null
  Invoke-WebRequest $url -OutFile "$dir\SKILL.md"
}
```

macOS:

```sh
for agent in ~/.claude ~/.codex; do
  [ -d "$agent" ] || continue   # that agent is not installed
  dir="$agent/skills/podmuncher"
  mkdir -p "$dir"
  curl -fsSL https://github.com/malindarathnayake/PodMuncher/releases/latest/download/podmuncher-SKILL.md -o "$dir/SKILL.md"
done
```

The snippets skip an agent whose home folder (`~/.claude`, `~/.codex`) does not exist.
Tell the user they can now ask, for example, "transcribe this episode" with a file or link.

## Upgrading

Repeat step 3 (it installs the latest release over the old one), then run
`podmuncher service stop` and step 5 again so the new images are pulled.
