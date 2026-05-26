# reel2md

`reel2md` is a local-first CLI that turns a reel or short-form video URL into a Markdown file that is easier to feed into AI tools later.

This tool can be useful for technical reels, tutorials, commentary, cooking, travel, maker content, fitness, and other short-form videos. That said, it works best as an extraction and note-generation tool, not as a perfect reconstruction tool. Reels are a difficult source format: tiny text, fast cuts, heavy compression, and incomplete narration all limit what any local pipeline can recover.

It also works with YouTube Shorts and regular YouTube videos because the ingestion layer is `yt-dlp`, not an Instagram-specific scraper. Shorts are a natural fit. Longer YouTube videos are supported too, but they are more expensive to process and should usually be run with tighter frame budgets such as `--speed-mode fast` or `--max-frames N`.

## What It Does

At a high level, `reel2md`:

1. Resolves the URL with `yt-dlp` metadata.
2. Chooses a title and output filename.
3. Downloads the video locally.
4. Extracts candidate frames using scene detection, with a fixed-interval fallback.
5. Transcribes the audio with Whisper.
6. Selects a capped subset of frames for deeper analysis, so runtime does not scale linearly with every cut.
7. Runs OCR with Tesseract.
8. Runs local multimodal analysis through Ollama for `Visual` and `On-screen text`.
9. Merges transcript, visuals, and detected text into one Markdown file.

## What The Output Is Good For

- Building a personal archive of reels in a text format.
- Feeding a reel into ChatGPT, Claude, Gemini, or local agents without uploading video.
- Searching for rough steps, components, topics, or spoken explanations later.
- Creating "memory aid" notes for technical or instructional reels.

## What The Output Is Not Good For

- Recovering exact source code from a tiny screen recording.
- Extracting every pin label, schematic label, or subtitle perfectly for detailed reconstruction of a project.
- Reconstructing every visual beat from a fast montage reel.
- Replacing a full tutorial, repo, blog post, or long-form source if one exists.

## Current Tradeoff

The tool is deliberately optimized for "high-signal notes" rather than "perfect exhaustive extraction."

That means:

- Transcript is usually the strongest signal.
- Visual descriptions are helpful, but imperfect.
- On-screen text is the hardest part and still the noisiest part.
- Very dense reels must be budgeted, otherwise runtime becomes unreasonable.

## Requirements

You need these system tools on your `PATH`:

- `ffmpeg`
- `tesseract`
- `ollama`

You also need these Python packages, which are installed by this project:

- `yt-dlp`
- `openai-whisper`
- `requests`
- `pytesseract`
- `Pillow`

## Windows Setup

These are the practical setup steps for a Windows machine.

1. Install the package from this folder:

```powershell
pip install -e .
```

2. Install Ollama if it is missing:

```powershell
irm https://ollama.com/install.ps1 | iex
```

3. Pull a vision model after installing Ollama:

```powershell
ollama pull llava
```

4. Install FFmpeg from an elevated terminal if it is missing:

```powershell
winget install ffmpeg
```

5. Install Tesseract OCR.

6. Add the Tesseract install directory to `PATH`.

7. Close and reopen your terminal after changing `PATH`.

8. Run the dependency check:

```powershell
reel2md doctor
```

## Installation

From the project folder:

```powershell
pip install -e .
```

If you already installed in editable mode, later code changes are picked up automatically.

## Default Output Directory

By default, generated Markdown files go to:

```text
D:\Projects\Ideas
```

This default is intentionally easy to change for other machines.

Override it per-command with:

```powershell
reel2md "URL" --output-dir "C:\somewhere\else"
```

Or override it through an environment variable:

```powershell
$env:REEL2MD_OUTPUT_DIR="C:\somewhere\else"
```

## Basic Usage

Run the dependency check:

```powershell
reel2md doctor
```

Convert a reel:

```powershell
reel2md "https://www.instagram.com/reel/..."
```

Convert a YouTube Short:

```powershell
reel2md "https://www.youtube.com/shorts/VIDEO_ID" --speed-mode fast
```

Convert a regular YouTube video:

```powershell
reel2md "https://www.youtube.com/watch?v=VIDEO_ID" --speed-mode fast --max-frames 8
```

Convert a reel that needs browser cookies:

```powershell
reel2md "https://www.instagram.com/reel/..." --cookies-from-browser chrome
```

Force regeneration of an already-processed reel:

```powershell
reel2md "https://www.instagram.com/reel/..." --force
```

Keep temporary files for debugging:

```powershell
reel2md "https://www.instagram.com/reel/..." --keep-temp --force
```

## Speed Modes

Runtime is dominated by OCR plus especially the Ollama vision stage. Dense reels with lots of cuts can become very slow if every candidate frame is analyzed.

To deal with that, `reel2md` now uses frame budgets and speed modes.

### `fast`

- Lowest runtime.
- Small adaptive frame budget.
- Good for quick note extraction.
- Best default when you care more about speed than coverage.

### `balanced`

- Default mode.
- Middle-ground frame budget.
- Good for most reels.

### `dense`

- Higher frame budget.
- Better for instructional reels that change scenes often.
- Slower than `balanced`.

### `max`

- Highest frame budget.
- Slowest mode.
- Use when you explicitly want more visual coverage and are okay paying for it in runtime.

Example:

```powershell
reel2md "https://www.instagram.com/reel/..." --speed-mode fast
```

## Frame Budget Override

If you want exact control, use `--max-frames`.

Example:

```powershell
reel2md "https://www.instagram.com/reel/..." --max-frames 6
```

This is a hard cap on how many frames are sent to the expensive OCR and vision stages.

If you do not pass `--max-frames`, the selected frame count is adaptive within the chosen speed mode. In other words, `fast` does not always mean exactly 6 frames. It means "at most 6 frames" with a smaller count used when the reel looks simpler.

## YouTube Notes

- YouTube Shorts work similarly to Instagram reels.
- Regular YouTube videos are supported, but long-form content is much more likely to be slow if you do not cap frames.
- Very high-resolution videos, high-frame-rate videos, and AV1 videos can be heavier for FFmpeg and slower overall.
- If scene detection finds no usable frames for a long-form or high-resolution video, the pipeline now falls back to fixed-interval extraction instead of aborting the run.

## CLI Options

Main options:

```text
--output-dir PATH
--vision-model NAME
--whisper-model NAME
--ollama-host URL
--scene-threshold FLOAT
--frame-workers N
--speed-mode {fast,balanced,dense,max}
--max-frames N
--cookies-from-browser NAME
--skip-ocr
--skip-vision
--skip-transcription
--keep-temp
--force
```

What they mean:

- `--output-dir`: where the Markdown file is written.
- `--vision-model`: local Ollama vision model name, for example `llava`.
- `--whisper-model`: Whisper model size such as `base` or `small`.
- `--ollama-host`: Ollama server URL, usually `http://127.0.0.1:11434`.
- `--scene-threshold`: FFmpeg scene-change threshold.
- `--frame-workers`: parallelism for frame OCR and vision.
- `--speed-mode`: runtime/quality preset.
- `--max-frames`: hard budget for expensive frame analysis.
- `--cookies-from-browser`: lets `yt-dlp` reuse browser cookies, useful for Instagram.
- `--skip-ocr`: disables OCR.
- `--skip-vision`: disables Ollama image analysis.
- `--skip-transcription`: disables Whisper transcription.
- `--keep-temp`: preserves downloads and extracted frames for inspection.
- `--force`: regenerates output for an already-known reel.

## How Deduplication Works

Filenames no longer need the platform video id in the name.

Instead:

- Output filenames are title-based.
- The tool keeps a small local index file in the output directory.
- Deduplication is based on reel identity, not just filename text.

This means you can have cleaner filenames without losing the ability to detect already-processed videos.

## Example Output

Each run writes one `.md` file with frontmatter and timestamped steps.

Example:

```md
---
source: https://www.instagram.com/reel/...
platform: instagram
video_id: Cx123Abc
title: Example Reel
duration_seconds: 42
processed_at: 2026-05-26T23:00:00+05:30
---

# Example Reel

## Source Description

Original caption or description text.

## Steps

**[00:00-00:05]**
- Spoken: First connect VCC to 3.3 volts.
- Visual: Close-up of an ESP32 dev board wired to a sensor module.
- On-screen text: GPIO 34 | VCC | GND
```

## Current Pipeline Notes

- Transcript usually carries the backbone of the note.
- Visual descriptions are generated by a local Ollama vision model.
- On-screen text is currently a hybrid of Tesseract OCR plus a vision-model text guess.
- The tool prefers extracted evidence and avoids adding an extra summarization LLM pass after the pipeline.

## Practical Expectations

Reels vary a lot in how extractable they are.

Reels that work better:

- clear narration
- fewer cuts
- larger on-screen text
- simpler layouts
- slower pacing

Reels that work worse:

- dense code on a tiny laptop screen
- rapid montages
- text overlays that appear for a split second
- heavy compression
- stylized or animated captions

In practice:

- Simple reels can produce very solid Markdown.
- Technical reels can produce very useful notes, but not perfect technical documentation.
- Fast montage reels are best treated as "high-level memory aids," not exhaustive structured references.

## Troubleshooting

### `reel2md doctor` says `ffmpeg` is missing

Install FFmpeg and ensure it is on `PATH`.

### `reel2md doctor` says `tesseract` is missing

Install Tesseract and ensure its install directory is on `PATH`.

### `reel2md doctor` says Ollama is unreachable

Make sure Ollama is installed and running.

### Instagram extraction fails or returns restricted content

Try:

```powershell
reel2md "https://www.instagram.com/reel/..." --cookies-from-browser chrome
```

### It is taking too long

Try:

```powershell
reel2md "https://www.instagram.com/reel/..." --speed-mode fast
```

Or:

```powershell
reel2md "https://www.instagram.com/reel/..." --max-frames 6
```

The tool now chooses frame counts adaptively under each speed mode, so the same speed mode can analyze fewer frames on simpler reels and more frames on denser ones, up to the mode's cap.

### Output is transcript-heavy and light on visuals

That usually means the selected frames were not very informative, the reel is visually dense, or the budget was too low. Try `dense` mode or a slightly higher frame cap.

### On-screen text is noisy

This is currently the weakest part of the pipeline. OCR on short-form video frames is inherently difficult.

## Future Directions

Things that could still improve the pipeline:

- better text-region detection before OCR
- better downscaling and preprocessing heuristics
- faster or more accurate vision models
- caching OCR/vision results across reruns
- optional summary or metadata passes
- optional domain modes such as `hardware`, `coding`, or `general`

## Notes

- The CLI keeps the output focused on extracted evidence. It does not invent a bill of materials or do a second expensive summary pass by default.
- If you want an Obsidian or RAG layer later, that can sit on top of this output without changing the ingestion pipeline.
- If scene-change parsing is weak for a given reel, the CLI falls back to fixed-interval frame extraction.
- Dense reels no longer scale linearly with every cut because the pipeline now ranks and caps frames before expensive analysis.
