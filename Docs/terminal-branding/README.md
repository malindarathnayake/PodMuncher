# Podmuncher terminal mascot study

Design-time assets only; no application integration has been made.

- `ascii/podmuncher-classic-28.txt`: portable ASCII fallback, 28 columns by 14 rows.
- `ascii/podmuncher-block-28.txt`: optional Unicode block variant, 28 columns by 14 rows; preferred where block glyphs have consistent monospace widths.
- `terminal-mascot-preview.png`: both actual TXT files rendered in Consolas on the kit's Night background.
- `render-preview.py`: reproduces the preview and checks dimensions, ASCII fallback, and absence of ANSI escapes. Uses Pillow from the converter's isolated environment.

The open mouth, eye, and two adjacent bars remain recognizable at this size. The classic fallback has more visual noise from the density ramp; the block variant reads more clearly. Antialiasing from conversion produces a slight fringe. These are decorative candidates for an about or first-run screen, not normal repeated command output. TXT has no embedded colors and inherits the terminal foreground/background. Preserve spaces and line breaks. Unicode display quality depends on the terminal font; use the classic fallback when uncertain.

## Source and provenance

Artwork source: `../Podmuncher branding kit/assets/png/podmuncher-symbol-mono-white-512x512.png` (the kit's monochrome character symbol). The standalone silhouette was inspected first, but its dark fill is unsuitable for this converter's fixed black compositing; the white symbol avoids modifying the supplied artwork.

Source SHA-256: `1C6236CC7A2A4D0B03BE9B7BD3E222FD57C5026D363D8FC6047F03A549C872E2`.

Artwork licensing evidence: `../Podmuncher branding kit/assets/README.md` describes the artwork as original and **suggests** CC BY 4.0, with a separate trademark policy. That is a proposed license, not a confirmed grant. These derivatives retain that unresolved artwork licensing status. No project ownership or trademark permission is inferred from converter licensing.

Converter: installed `ascii-art` skill at `~/.codex/skills/ascii-art/`, upstream URL recorded in its README: https://github.com/neethanwu/ascii-art. Local `LICENSE` is MIT, copyright (c) 2026 ascii-art contributors. No converter code is copied into this directory. The installed snapshot has no Git metadata or declared release version; its exact commit is unavailable. `scripts/convert.py` SHA-256: `D50944E7152B92333DD8A06D4A37A22F85084EC35172EED5D3F015675D50CCF3`.

Created 2026-10-09 using Python 3.12, Pillow 12.3.0, NumPy 2.5.3, and pyfiglet 1.0.4. Dependencies are isolated in the skill's `scripts/.venv`. The Bash setup expects Unix `bin/python`, so its required environment steps were adapted to Windows `Scripts/python.exe`. Optional OpenCV/video dependencies were omitted. No global packages, clipboard operations, or external uploads were used.

## Reproduce on this workspace

From `Docs/terminal-branding` in PowerShell:

```powershell
$converterPython = "$env:USERPROFILE\.codex\skills\ascii-art\scripts\.venv\Scripts\python.exe"
$converterScript = "$env:USERPROFILE\.codex\skills\ascii-art\scripts\convert.py"
$mascotSource = '..\Podmuncher branding kit\assets\png\podmuncher-symbol-mono-white-512x512.png'
& $converterPython $converterScript --input $mascotSource --type image --style classic --color grayscale --background dark --export txt --cols 28 --invert --filename podmuncher-classic-28
& $converterPython $converterScript --input $mascotSource --type image --style block --color grayscale --background dark --export txt --cols 28 --filename podmuncher-block-28
& $converterPython '.\render-preview.py'
```

The inversion flags intentionally differ: the converter's classic and block mappings have opposite density directions. Both exports use the default original aspect ratio and no dithering. The rendered preview was visually inspected, and both TXT files passed the documented structural checks.
