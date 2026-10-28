# Running Podmuncher on macOS

Podmuncher runs on macOS with Docker Desktop. It was validated on an Apple M4 (arm64)
running macOS 27 and Docker Desktop 4.64. The media and CPU transcription images are built
for both `linux/amd64` and `linux/arm64`, and Docker pulls the image that matches its engine.

## What you need

- Docker Desktop for Mac, installed and running. Setup never installs it for you.
- Python 3.12 or later. The system Python on macOS is too old; use one from python.org, Homebrew
  or `uv`.
- Enough free disk on the volume that holds Docker Desktop's disk image and on the volumes you
  choose for the cache, scratch space and output. `podmuncher setup` shows the plan per volume
  before it changes anything.

## Transcription runs on the CPU

Docker Desktop on a Mac cannot pass the Apple GPU to Linux containers, and the CUDA image is
built only for `linux/amd64`. Podmuncher therefore never tries the CUDA image on an arm64
engine. Transcription uses the CPU image, and in automatic mode the model is picked from the
memory that Docker Desktop gives its VM. To give larger models room, raise the memory in
**Docker Desktop → Settings → Resources**. Large models on the CPU are slow. That is your
choice to make: you can pick any pinned model explicitly during setup.

## Docker must be on your PATH

Docker Desktop installs its command-line tool at `/usr/local/bin/docker`. macOS puts
`/usr/local/bin` on the PATH of every Terminal shell. If you start Podmuncher some other way,
for example from a non-login SSH session or a launch agent, add `/usr/local/bin` to its PATH
first.

## Shared folders

Docker Desktop only lets containers mount folders listed under **Docker Desktop → Settings →
Resources → File sharing**. The default Podmuncher locations under your home folder (`/Users`)
are shared already. If you choose an output, cache or scratch folder somewhere else, for example
on an external drive or under `/opt`, setup warns you. Add that folder in File sharing, or choose a
folder inside one that is already listed. Setup does not block you; jobs that use an unshared
folder fail when the container starts.

## Docker Desktop disk space

Docker Desktop keeps its Linux VM disk in a sparse file at
`~/Library/Containers/com.docker.docker/Data/vms/0/data/Docker.raw`. Podmuncher reads the
disk limit from that file's size and treats the space allocated to it as used. It counts the
free space as the smaller of what is left under the limit and what is free on the Mac volume.
Space freed inside the VM may stay allocated in the file, so the reading can be lower than the
real free space; it is never higher. If you moved the disk image, enter its location when setup
asks for the Docker data location.

## Open-file limit

macOS starts processes with a soft limit of 256 open files. The Podmuncher service raises its
own soft limit to 8192, or to the hard limit if that is lower, when it starts. Nothing needs
to be done for normal use.

## Running the test suite on a Mac (developers)

- Raise the open-file limit in the shell first: `ulimit -n 10240`. The suite opens many
  files at once and fails with `EMFILE` at the default of 256.
- The console script is `.venv/bin/podmuncher`.
- Over SSH, the Docker credential helper cannot reach the login keychain. If image pulls fail
  with a keychain error, point `DOCKER_CONFIG` at a directory whose `config.json` does not
  use `credsStore: desktop`. Only do this for public images that need no login.
- Live tests need the same environment variables as on other platforms (see
  `Docs/testing-harness.md`). Use images built for `linux/arm64`.
