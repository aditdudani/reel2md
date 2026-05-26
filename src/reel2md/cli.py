from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .doctor import run_doctor
from .pipeline import PipelineOptions, Reel2MdError, run_pipeline


def build_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reel2md",
        description="Convert a reel or short-form video URL into a Markdown file.",
    )
    parser.add_argument("url", help="Reel or short-form video URL")
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("REEL2MD_OUTPUT_DIR", r"D:\Projects\Ideas"),
        help="Directory for generated Markdown files",
    )
    parser.add_argument(
        "--vision-model",
        default=os.environ.get("REEL2MD_VISION_MODEL", "llava"),
        help="Local Ollama vision model name",
    )
    parser.add_argument(
        "--whisper-model",
        default=os.environ.get("REEL2MD_WHISPER_MODEL", "base"),
        help="Whisper model size to use for transcription",
    )
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get("REEL2MD_OLLAMA_HOST", "http://127.0.0.1:11434"),
        help="Ollama base URL",
    )
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=float(os.environ.get("REEL2MD_SCENE_THRESHOLD", "0.35")),
        help="FFmpeg scene-change threshold",
    )
    parser.add_argument(
        "--frame-workers",
        type=int,
        default=int(os.environ.get("REEL2MD_FRAME_WORKERS", "1")),
        help="Number of worker threads for OCR and vision",
    )
    parser.add_argument(
        "--speed-mode",
        choices=["fast", "balanced", "dense", "max"],
        default=os.environ.get("REEL2MD_SPEED_MODE", "balanced"),
        help="Runtime/quality preset for frame selection",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=int(os.environ["REEL2MD_MAX_FRAMES"]) if os.environ.get("REEL2MD_MAX_FRAMES") else None,
        help="Hard cap on frames sent to OCR and vision",
    )
    parser.add_argument(
        "--cookies-from-browser",
        default=os.environ.get("REEL2MD_COOKIES_FROM_BROWSER"),
        help="Browser name for yt-dlp cookies, e.g. chrome or firefox",
    )
    parser.add_argument("--skip-ocr", action="store_true", help="Disable OCR on extracted frames")
    parser.add_argument(
        "--skip-vision",
        action="store_true",
        help="Disable local vision descriptions from Ollama",
    )
    parser.add_argument(
        "--skip-transcription",
        action="store_true",
        help="Disable Whisper speech transcription",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary download and frame files after the run",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing Markdown file for the same video",
    )
    return parser


def build_doctor_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reel2md doctor",
        description="Check local dependencies for reel2md.",
    )
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get("REEL2MD_OLLAMA_HOST", "http://127.0.0.1:11434"),
        help="Ollama base URL",
    )
    parser.add_argument(
        "--vision-model",
        default=os.environ.get("REEL2MD_VISION_MODEL", "llava"),
        help="Vision model name expected in Ollama",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "doctor":
        parser = build_doctor_parser()
        args = parser.parse_args(argv[1:])
        return run_doctor(args.ollama_host, args.vision_model)

    parser = build_run_parser()
    args = parser.parse_args(argv)
    options = PipelineOptions(
        output_dir=Path(args.output_dir),
        vision_model=args.vision_model,
        whisper_model=args.whisper_model,
        ollama_host=args.ollama_host,
        scene_threshold=args.scene_threshold,
        frame_workers=args.frame_workers,
        speed_mode=args.speed_mode,
        max_frames=args.max_frames,
        cookies_from_browser=args.cookies_from_browser,
        skip_ocr=args.skip_ocr,
        skip_vision=args.skip_vision,
        skip_transcription=args.skip_transcription,
        keep_temp=args.keep_temp,
        force=args.force,
    )

    try:
        output_path = run_pipeline(args.url, options)
    except Reel2MdError as exc:
        print(f"[!] {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("[!] Interrupted.", file=sys.stderr)
        return 130

    print(f"[*] Markdown written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
