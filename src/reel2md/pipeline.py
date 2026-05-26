from __future__ import annotations

import base64
import json
import math
import re
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".m4v"}
METADATA_HEADER_RE = re.compile(
    r"^frame\s+(?P<frame>\d+)\s+pts\s+(?P<pts>-?\d+)\s+pts_time\s+(?P<pts_time>-?[0-9.]+)\s*$"
)
SHOWINFO_RE = re.compile(r"pts_time:(?P<pts_time>-?[0-9.]+)")

SPEED_PROFILES = {
    "fast": {
        "max_frames": 6,
        "min_frames": 3,
        "max_frames_per_segment": 1,
        "max_visuals_per_segment": 1,
    },
    "balanced": {
        "max_frames": 10,
        "min_frames": 5,
        "max_frames_per_segment": 2,
        "max_visuals_per_segment": 2,
    },
    "dense": {
        "max_frames": 18,
        "min_frames": 8,
        "max_frames_per_segment": 2,
        "max_visuals_per_segment": 2,
    },
    "max": {
        "max_frames": 28,
        "min_frames": 12,
        "max_frames_per_segment": 3,
        "max_visuals_per_segment": 3,
    },
}


class Reel2MdError(RuntimeError):
    """Raised when the pipeline cannot continue."""


@dataclass
class MediaInfo:
    source_url: str
    canonical_url: str
    platform: str
    video_id: str
    title: str
    uploader: str
    description: str
    duration_seconds: Optional[int]
    media_path: Optional[Path] = None
    dedupe_key: Optional[str] = None


@dataclass
class FrameRecord:
    timestamp_seconds: float
    image_path: Path
    scene_score: Optional[float] = None
    visual: str = ""
    ocr_text: str = ""
    ocr_score: float = 0.0
    vision_text: str = ""


@dataclass
class TranscriptSegment:
    start_seconds: float
    end_seconds: float
    spoken: str


@dataclass
class MergedStep:
    start_seconds: float
    end_seconds: float
    spoken: str
    visuals: List[str]
    on_screen_text: List[str]


@dataclass
class PipelineOptions:
    output_dir: Path
    vision_model: str
    whisper_model: str
    ollama_host: str
    scene_threshold: float
    frame_workers: int
    speed_mode: str = "balanced"
    max_frames: Optional[int] = None
    cookies_from_browser: Optional[str] = None
    skip_ocr: bool = False
    skip_vision: bool = False
    skip_transcription: bool = False
    keep_temp: bool = False
    force: bool = False


def run_pipeline(url: str, options: PipelineOptions) -> Path:
    output_dir = options.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    media_info = inspect_media(url, cookies_from_browser=options.cookies_from_browser)
    index = load_output_index(output_dir)
    existing_output = find_existing_output_path(output_dir, media_info, index=index)
    output_path = existing_output or build_output_path(output_dir, media_info, index=index)
    if existing_output and not options.force:
        raise Reel2MdError(
            f"Markdown already exists at {existing_output}. Re-run with --force to regenerate."
        )

    if options.skip_ocr and options.skip_vision and options.skip_transcription:
        raise Reel2MdError("At least one modality must remain enabled.")

    ensure_command("ffmpeg")
    if not options.skip_ocr:
        ensure_command("tesseract")

    temp_dir = Path(tempfile.mkdtemp(prefix="reel2md_"))
    try:
        downloaded = download_media(
            media_info,
            temp_dir,
            cookies_from_browser=options.cookies_from_browser,
        )

        frames = extract_scene_frames(
            downloaded.media_path,
            temp_dir / "frames",
            scene_threshold=options.scene_threshold,
        )
        print(f"[*] Extracted {len(frames)} candidate frames")

        transcript_segments: List[TranscriptSegment] = []
        if not options.skip_transcription:
            transcript_segments = transcribe_media(
                downloaded.media_path, whisper_model=options.whisper_model
            )
            print(f"[*] Transcribed {len(transcript_segments)} spoken segments")

        selected_frames = select_frames_for_analysis(
            frames,
            transcript_segments,
            options=options,
        )
        if len(selected_frames) != len(frames):
            print(f"[*] Selected {len(selected_frames)} of {len(frames)} frames for OCR/vision")
        frames = selected_frames

        if frames and (not options.skip_ocr or not options.skip_vision):
            frames = enrich_frames(
                frames,
                transcript_segments=transcript_segments,
                ollama_host=options.ollama_host,
                vision_model=options.vision_model,
                skip_ocr=options.skip_ocr,
                skip_vision=options.skip_vision,
                frame_workers=options.frame_workers,
            )
            print(f"[*] Enriched {len(frames)} frames with OCR/vision")

        merged_steps = merge_modalities(
            transcript_segments,
            frames,
            max_visuals_per_segment=resolve_speed_profile(options)["max_visuals_per_segment"],
        )
        markdown = render_markdown(
            media=downloaded,
            steps=merged_steps,
            options=options,
        )
        output_path.write_text(markdown, encoding="utf-8")
        update_output_index(output_dir, index=index, media=downloaded, output_path=output_path)
        return output_path
    finally:
        if options.keep_temp:
            print(f"[*] Kept temp directory: {temp_dir}")
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)


def inspect_media(url: str, cookies_from_browser: Optional[str] = None) -> MediaInfo:
    yt_dlp = import_yt_dlp()
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }
    if cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    platform = normalize_platform(info.get("extractor_key") or info.get("extractor") or "video")
    video_id = str(info.get("id") or stable_hash(url))
    return MediaInfo(
        source_url=url,
        canonical_url=info.get("webpage_url") or url,
        platform=platform,
        video_id=video_id,
        title=choose_title(
            raw_title=info.get("title") or "",
            description=info.get("description") or "",
            uploader=info.get("uploader") or info.get("channel") or "",
            platform=platform,
            video_id=video_id,
        ),
        uploader=clean_text(info.get("uploader") or info.get("channel") or ""),
        description=clean_text(info.get("description") or ""),
        duration_seconds=safe_int(info.get("duration")),
        dedupe_key=build_dedupe_key(
            platform=platform,
            video_id=video_id,
            source_url=info.get("webpage_url") or url,
        ),
    )


def download_media(
    media: MediaInfo,
    temp_dir: Path,
    cookies_from_browser: Optional[str] = None,
) -> MediaInfo:
    yt_dlp = import_yt_dlp()
    download_dir = temp_dir / "download"
    download_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(download_dir / f"{media.platform}_{media.video_id}.%(ext)s")
    ydl_opts = {
        "quiet": False,
        "no_warnings": True,
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
        "merge_output_format": "mp4",
        "outtmpl": outtmpl,
        "noprogress": True,
    }
    if cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(media.source_url, download=True)

    media_path = find_downloaded_media(download_dir)
    media.media_path = media_path
    return media


def extract_scene_frames(
    media_path: Path,
    frames_dir: Path,
    scene_threshold: float,
) -> List[FrameRecord]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = frames_dir / "frame_%05d.jpg"
    filter_expr = f"select='gt(scene,{scene_threshold})',showinfo"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(media_path),
        "-vf",
        filter_expr,
        "-fps_mode",
        "vfr",
        str(output_pattern),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    metadata_entries = parse_showinfo_output(result.stderr)

    if result.returncode != 0 and should_fallback_scene_extraction(result.stderr, frame_files, metadata_entries):
        print("[*] Scene detection produced no usable frames, falling back to fixed-interval extraction")
        return extract_fixed_interval_frames(media_path, frames_dir)

    if result.returncode != 0:
        raise Reel2MdError(f"ffmpeg scene extraction failed:\n{result.stderr.strip()}")

    if not frame_files or not metadata_entries:
        return extract_fixed_interval_frames(media_path, frames_dir)

    if len(frame_files) != len(metadata_entries):
        paired = min(len(frame_files), len(metadata_entries))
        frame_files = frame_files[:paired]
        metadata_entries = metadata_entries[:paired]

    records = []
    for image_path, metadata in zip(frame_files, metadata_entries):
        records.append(
            FrameRecord(
                timestamp_seconds=metadata["pts_time"],
                image_path=image_path,
                scene_score=metadata.get("scene_score"),
            )
        )
    return records


def extract_fixed_interval_frames(media_path: Path, frames_dir: Path) -> List[FrameRecord]:
    for old_frame in frames_dir.glob("frame_*.jpg"):
        old_frame.unlink()

    output_pattern = frames_dir / "frame_%05d.jpg"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(media_path),
        "-vf",
        "fps=1/5",
        "-fps_mode",
        "vfr",
        str(output_pattern),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise Reel2MdError(f"ffmpeg fallback extraction failed:\n{result.stderr.strip()}")

    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    records = []
    for index, image_path in enumerate(frame_files):
        records.append(
            FrameRecord(
                timestamp_seconds=float(index * 5),
                image_path=image_path,
            )
        )
    return records


def enrich_frames(
    frames: Sequence[FrameRecord],
    *,
    transcript_segments: Sequence[TranscriptSegment],
    ollama_host: str,
    vision_model: str,
    skip_ocr: bool,
    skip_vision: bool,
    frame_workers: int,
) -> List[FrameRecord]:
    if frame_workers < 1:
        frame_workers = 1

    if not skip_vision:
        ensure_ollama_model_available(ollama_host, vision_model)

    if frame_workers == 1:
        return [
            enrich_single_frame(
                frame,
                transcript_segments=transcript_segments,
                ollama_host=ollama_host,
                vision_model=vision_model,
                skip_ocr=skip_ocr,
                skip_vision=skip_vision,
            )
            for frame in frames
        ]

    from concurrent.futures import ThreadPoolExecutor

    results: List[Optional[FrameRecord]] = [None] * len(frames)
    with ThreadPoolExecutor(max_workers=frame_workers) as pool:
        futures = []
        for index, frame in enumerate(frames):
            futures.append(
                (
                    index,
                    pool.submit(
                        enrich_single_frame,
                        frame,
                        transcript_segments,
                        ollama_host=ollama_host,
                        vision_model=vision_model,
                        skip_ocr=skip_ocr,
                        skip_vision=skip_vision,
                    ),
                )
            )
        for index, future in futures:
            results[index] = future.result()

    return [frame for frame in results if frame is not None]


def enrich_single_frame(
    frame: FrameRecord,
    transcript_segments: Sequence[TranscriptSegment],
    *,
    ollama_host: str,
    vision_model: str,
    skip_ocr: bool,
    skip_vision: bool,
) -> FrameRecord:
    if not skip_ocr:
        frame.ocr_text, frame.ocr_score = read_ocr_text(frame.image_path)
    if not skip_vision:
        frame.visual, frame.vision_text = analyze_frame_with_ollama(
            frame.image_path,
            ollama_host=ollama_host,
            model=vision_model,
            ocr_hint=frame.ocr_text,
            transcript_hint=find_transcript_hint(frame.timestamp_seconds, transcript_segments),
        )
        frame.ocr_text = choose_on_screen_text(
            tesseract_text=frame.ocr_text,
            tesseract_score=frame.ocr_score,
            vision_text=frame.vision_text,
        )
    return frame


def transcribe_media(media_path: Path, whisper_model: str) -> List[TranscriptSegment]:
    whisper = import_whisper()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"TypedStorage is deprecated\..*",
            category=UserWarning,
        )
        model = whisper.load_model(whisper_model)
        result = model.transcribe(str(media_path), fp16=False, verbose=False)
    segments = []
    for raw in result.get("segments", []):
        spoken = clean_text(raw.get("text") or "")
        if not spoken:
            continue
        segments.append(
            TranscriptSegment(
                start_seconds=float(raw["start"]),
                end_seconds=float(raw["end"]),
                spoken=spoken,
            )
        )
    return segments


def merge_modalities(
    transcript_segments: Sequence[TranscriptSegment],
    frames: Sequence[FrameRecord],
    *,
    max_visuals_per_segment: int = 2,
) -> List[MergedStep]:
    if not transcript_segments:
        return merge_without_transcript(frames)

    steps = [
        MergedStep(
            start_seconds=segment.start_seconds,
            end_seconds=segment.end_seconds,
            spoken=segment.spoken,
            visuals=[],
            on_screen_text=[],
        )
        for segment in transcript_segments
    ]

    assigned_frames: Dict[int, List[FrameRecord]] = {index: [] for index in range(len(steps))}
    for frame in frames:
        step_index = find_nearest_step_index(steps, frame.timestamp_seconds)
        if step_index is not None:
            assigned_frames[step_index].append(frame)

    for index, step in enumerate(steps):
        chosen_frames = choose_representative_frames(
            step,
            assigned_frames.get(index, []),
            limit=max_visuals_per_segment,
        )
        for frame in chosen_frames:
            append_frame_to_step(step, frame)

    return normalize_steps(steps)


def merge_without_transcript(frames: Sequence[FrameRecord]) -> List[MergedStep]:
    steps = []
    for frame in frames:
        steps.append(
            MergedStep(
                start_seconds=frame.timestamp_seconds,
                end_seconds=frame.timestamp_seconds,
                spoken="",
                visuals=[frame.visual] if frame.visual else [],
                on_screen_text=[frame.ocr_text] if frame.ocr_text else [],
            )
        )
    return normalize_steps(steps)


def render_markdown(
    *,
    media: MediaInfo,
    steps: Sequence[MergedStep],
    options: PipelineOptions,
) -> str:
    frontmatter = {
        "source": media.source_url,
        "canonical_url": media.canonical_url,
        "platform": media.platform,
        "video_id": media.video_id,
        "title": media.title,
        "uploader": media.uploader or None,
        "duration_seconds": media.duration_seconds,
        "processed_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "scene_threshold": options.scene_threshold,
        "vision_model": None if options.skip_vision else options.vision_model,
        "whisper_model": None if options.skip_transcription else options.whisper_model,
        "ocr_enabled": not options.skip_ocr,
    }

    lines = ["---", render_frontmatter(frontmatter), "---", ""]
    lines.append(f"# {media.title}")
    lines.append("")

    if media.description:
        lines.append("## Source Description")
        lines.append("")
        lines.append(media.description)
        lines.append("")

    lines.append("## Steps")
    lines.append("")

    if not steps:
        lines.append("No transcript segments or frame insights were extracted.")
        lines.append("")
        return "\n".join(lines)

    for step in steps:
        lines.append(f"**[{format_timestamp(step.start_seconds)}-{format_timestamp(step.end_seconds)}]**")
        if step.spoken:
            lines.append(f"- Spoken: {step.spoken}")
        else:
            lines.append("- Spoken: ")

        if step.visuals:
            for visual in step.visuals:
                lines.append(f"- Visual: {visual}")
        else:
            lines.append("- Visual: ")

        if step.on_screen_text:
            for text in step.on_screen_text:
                lines.append(f"- On-screen text: {text}")
        else:
            lines.append("- On-screen text: ")

        lines.append("")

    return "\n".join(lines)


def build_output_path(output_dir: Path, media: MediaInfo, *, index: Dict[str, str]) -> Path:
    stem = slugify_filename(media.title) or slugify_filename(media.platform) or "reel"
    candidate = output_dir / f"{stem}.md"
    suffix = 2
    while candidate.exists() and not matches_index_path(index, media, candidate):
        candidate = output_dir / f"{stem}-{suffix}.md"
        suffix += 1
    return candidate


def find_existing_output_path(output_dir: Path, media: MediaInfo, *, index: Dict[str, str]) -> Optional[Path]:
    if media.dedupe_key and media.dedupe_key in index:
        candidate = output_dir / index[media.dedupe_key]
        if candidate.exists():
            return candidate
    return None


def render_frontmatter(data: Dict[str, object]) -> str:
    lines = []
    for key, value in data.items():
        if value is None:
            continue
        lines.append(f"{key}: {format_frontmatter_value(value)}")
    return "\n".join(lines)


def format_frontmatter_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "":
        return '""'
    if re.search(r"[:#\-\n\[\]\{\},]|^\s|\s$", text):
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return text


def ensure_command(name: str) -> None:
    if shutil.which(name):
        return
    raise Reel2MdError(
        f"Required command '{name}' was not found on PATH. Run 'reel2md doctor' for setup checks."
    )


def ensure_ollama_model_available(ollama_host: str, model: str) -> None:
    requests = import_requests()
    try:
        response = requests.get(f"{ollama_host.rstrip('/')}/api/tags", timeout=10)
        response.raise_for_status()
    except Exception as exc:  # pragma: no cover - network-style failure path
        raise Reel2MdError(
            f"Could not reach Ollama at {ollama_host}. Make sure Ollama is running."
        ) from exc

    payload = response.json()
    available = {item.get("name", "") for item in payload.get("models", [])}
    normalized_available = {name.split(":")[0] for name in available} | available
    if model not in normalized_available:
        raise Reel2MdError(
            f"Ollama model '{model}' is not available. Pull it first, for example: ollama pull {model}"
        )


def analyze_frame_with_ollama(
    image_path: Path,
    *,
    ollama_host: str,
    model: str,
    ocr_hint: str,
    transcript_hint: str,
) -> tuple[str, str]:
    requests = import_requests()
    prompt = (
        "Analyze this short-form video frame and return strict JSON with keys "
        "\"visual\" and \"on_screen_text\". "
        "\"visual\" should be one concise sentence about the important visible action or objects. "
        "\"on_screen_text\" should contain only clearly visible text that actually appears in the frame. "
        "Do not infer or paraphrase on-screen text from the transcript. "
        "If no readable on-screen text is visible, use an empty string."
    )
    if transcript_hint:
        prompt += f" Nearby transcript for visual context only: {transcript_hint}"
    if ocr_hint:
        prompt += f" OCR hint from the same frame that may help you read blurry text: {ocr_hint}"

    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    response = requests.post(
        f"{ollama_host.rstrip('/')}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    raw = payload.get("response") or ""
    parsed = parse_vision_response(raw)
    return parsed["visual"], parsed["on_screen_text"]


def read_ocr_text(image_path: Path) -> tuple[str, float]:
    pytesseract = import_pytesseract()
    pil_image_module = import_pil_image()
    with pil_image_module.open(image_path) as image:
        best_text = ""
        best_score = 0.0
        for variant in iter_ocr_variants(image):
            text, score = extract_ocr_from_variant(pytesseract, variant)
            if score > best_score:
                best_text = text
                best_score = score
    return best_text, best_score


def parse_ffmpeg_metadata_output(text: str) -> List[Dict[str, float]]:
    entries: List[Dict[str, float]] = []
    current: Optional[Dict[str, float]] = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        header_match = METADATA_HEADER_RE.match(line)
        if header_match:
            if current is not None:
                entries.append(current)
            current = {
                "frame": float(header_match.group("frame")),
                "pts": float(header_match.group("pts")),
                "pts_time": float(header_match.group("pts_time")),
            }
            continue
        if current is None or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key == "lavfi.scene_score":
            try:
                current["scene_score"] = float(value)
            except ValueError:
                pass
    if current is not None:
        entries.append(current)
    return entries


def parse_showinfo_output(text: str) -> List[Dict[str, float]]:
    entries: List[Dict[str, float]] = []
    for line in text.splitlines():
        if "showinfo" not in line or "pts_time:" not in line:
            continue
        match = SHOWINFO_RE.search(line)
        if not match:
            continue
        entries.append({"pts_time": float(match.group("pts_time"))})
    return entries


def should_fallback_scene_extraction(
    stderr_text: str,
    frame_files: Sequence[Path],
    metadata_entries: Sequence[Dict[str, float]],
) -> bool:
    if frame_files and metadata_entries:
        return False
    signals = (
        "No filtered frames for output stream",
        "Nothing was written into output file",
        "Could not open encoder before EOF",
        "Conversion failed!",
    )
    return any(signal in stderr_text for signal in signals)


def find_downloaded_media(download_dir: Path) -> Path:
    candidates = [path for path in download_dir.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS]
    if not candidates:
        raise Reel2MdError(
            f"No downloaded media file was found in {download_dir}. Check your URL and authentication settings."
        )
    candidates.sort(key=lambda path: (path.stat().st_mtime, path.stat().st_size), reverse=True)
    return candidates[0]


def append_frame_to_step(step: MergedStep, frame: FrameRecord) -> None:
    if frame.visual:
        step.visuals.append(frame.visual)
    if frame.ocr_text:
        step.on_screen_text.append(frame.ocr_text)


def find_nearest_step_index(steps: Sequence[MergedStep], timestamp_seconds: float) -> Optional[int]:
    if not steps:
        return None
    best_index = 0
    best_distance = None
    for index, step in enumerate(steps):
        if step.start_seconds <= timestamp_seconds <= step.end_seconds:
            return index
        distance = min(
            abs(timestamp_seconds - step.start_seconds),
            abs(timestamp_seconds - step.end_seconds),
        )
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_index = index
    return best_index


def normalize_steps(steps: Sequence[MergedStep]) -> List[MergedStep]:
    normalized = []
    for step in steps:
        normalized.append(
            MergedStep(
                start_seconds=step.start_seconds,
                end_seconds=max(step.start_seconds, step.end_seconds),
                spoken=clean_text(step.spoken),
                visuals=dedupe_preserve_order(clean_text(item) for item in step.visuals if clean_text(item)),
                on_screen_text=dedupe_preserve_order(
                    clean_text(item) for item in step.on_screen_text if clean_text(item)
                ),
            )
        )
    return normalized


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def format_timestamp(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def clean_text(value: str) -> str:
    value = value or ""
    value = repair_common_mojibake(value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def normalize_platform(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    return value or "video"


def stable_hash(value: str) -> str:
    import hashlib

    return hashlib.md5(value.encode("utf-8")).hexdigest()[:12]


def safe_int(value: object) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def import_yt_dlp():
    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover - import guard
        raise Reel2MdError("Python package 'yt-dlp' is not installed.") from exc
    return yt_dlp


def import_whisper():
    try:
        import whisper
    except ImportError as exc:  # pragma: no cover - import guard
        raise Reel2MdError("Python package 'openai-whisper' is not installed.") from exc
    return whisper


def import_pytesseract():
    try:
        import pytesseract
    except ImportError as exc:  # pragma: no cover - import guard
        raise Reel2MdError("Python package 'pytesseract' is not installed.") from exc
    return pytesseract


def import_pil_image():
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - import guard
        raise Reel2MdError("Python package 'Pillow' is not installed.") from exc
    return Image


def import_requests():
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - import guard
        raise Reel2MdError("Python package 'requests' is not installed.") from exc
    return requests


def slugify_filename(value: str) -> str:
    value = clean_text(value).lower()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    if value in {"", "untitled"}:
        return ""
    return value[:80]


def choose_title(
    *,
    raw_title: str,
    description: str,
    uploader: str,
    platform: str,
    video_id: str,
) -> str:
    title = clean_text(raw_title)
    description = clean_text(description)
    generic_title = not title or bool(re.fullmatch(r"(video|reel)( by .+)?", title.lower()))
    if generic_title and description:
        candidate = description.split("#", 1)[0]
        candidate = re.split(r"[.!?]", candidate)[0]
        candidate = clean_text(candidate)
        words = candidate.split()
        if words:
            return " ".join(words[:12])
    if generic_title and uploader:
        return f"{clean_text(uploader)} {platform}"
    return title or platform


def build_dedupe_key(*, platform: str, video_id: str, source_url: str) -> str:
    return f"{platform}:{video_id or stable_hash(source_url)}"


def load_output_index(output_dir: Path) -> Dict[str, str]:
    index_path = output_dir / ".reel2md-index.json"
    if not index_path.exists():
        return {}
    try:
        return json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def update_output_index(
    output_dir: Path,
    *,
    index: Dict[str, str],
    media: MediaInfo,
    output_path: Path,
) -> None:
    if media.dedupe_key:
        index[media.dedupe_key] = output_path.name
    index_path = output_dir / ".reel2md-index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")


def matches_index_path(index: Dict[str, str], media: MediaInfo, candidate: Path) -> bool:
    if not media.dedupe_key:
        return False
    return index.get(media.dedupe_key) == candidate.name


def repair_common_mojibake(value: str) -> str:
    suspicious_markers = ("\u00e2", "\u00c3", "\u00f0\u0178", "\u00e2\u20ac")
    if not any(marker in value for marker in suspicious_markers):
        return value
    try:
        repaired = value.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")
    except Exception:
        return value
    return repaired or value


def find_transcript_hint(
    timestamp_seconds: float,
    transcript_segments: Sequence[TranscriptSegment],
) -> str:
    index = find_nearest_step_index(
        [
            MergedStep(
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                spoken=segment.spoken,
                visuals=[],
                on_screen_text=[],
            )
            for segment in transcript_segments
        ],
        timestamp_seconds,
    )
    if index is None:
        return ""
    return transcript_segments[index].spoken


def select_frames_for_analysis(
    frames: Sequence[FrameRecord],
    transcript_segments: Sequence[TranscriptSegment],
    *,
    options: PipelineOptions,
) -> List[FrameRecord]:
    if not frames:
        return []

    profile = resolve_speed_profile(options)
    max_frames = compute_adaptive_frame_budget(
        frames,
        transcript_segments,
        options=options,
        profile=profile,
    )
    max_frames_per_segment = profile["max_frames_per_segment"]
    if max_frames is None or len(frames) <= max_frames:
        return list(frames)

    if not transcript_segments:
        return select_frames_without_transcript(frames, max_frames=max_frames)

    segment_steps = [
        MergedStep(
            start_seconds=segment.start_seconds,
            end_seconds=segment.end_seconds,
            spoken=segment.spoken,
            visuals=[],
            on_screen_text=[],
        )
        for segment in transcript_segments
    ]

    buckets: Dict[int, List[FrameRecord]] = {index: [] for index in range(len(segment_steps))}
    for frame in frames:
        index = find_nearest_step_index(segment_steps, frame.timestamp_seconds)
        if index is not None:
            buckets[index].append(frame)

    ranked_buckets: List[tuple[int, List[FrameRecord], float]] = []
    for index, step in enumerate(segment_steps):
        ranked = sorted(
            buckets.get(index, []),
            key=lambda frame: rank_candidate_frame(frame, step, step.spoken),
            reverse=True,
        )[:max_frames_per_segment]
        if not ranked:
            continue
        ranked_buckets.append((index, ranked, rank_segment_importance(step.spoken)))

    ranked_buckets.sort(key=lambda item: item[2], reverse=True)
    selected: List[FrameRecord] = []
    used_paths = set()
    round_index = 0
    while len(selected) < max_frames:
        made_progress = False
        for _segment_index, ranked, _importance in ranked_buckets:
            if round_index >= len(ranked):
                continue
            frame = ranked[round_index]
            key = str(frame.image_path)
            if key in used_paths:
                continue
            selected.append(frame)
            used_paths.add(key)
            made_progress = True
            if len(selected) >= max_frames:
                break
        if not made_progress:
            break
        round_index += 1

    if len(selected) < max_frames:
        leftovers = sorted(
            frames,
            key=lambda frame: fallback_frame_priority(frame),
            reverse=True,
        )
        for frame in leftovers:
            key = str(frame.image_path)
            if key in used_paths:
                continue
            selected.append(frame)
            used_paths.add(key)
            if len(selected) >= max_frames:
                break

    selected.sort(key=lambda frame: frame.timestamp_seconds)
    return selected


def select_frames_without_transcript(
    frames: Sequence[FrameRecord],
    *,
    max_frames: int,
) -> List[FrameRecord]:
    ranked = sorted(frames, key=fallback_frame_priority, reverse=True)
    selected = ranked[:max_frames]
    selected.sort(key=lambda frame: frame.timestamp_seconds)
    return selected


def resolve_speed_profile(options: PipelineOptions) -> Dict[str, int]:
    profile = dict(SPEED_PROFILES.get(options.speed_mode, SPEED_PROFILES["balanced"]))
    if options.max_frames is not None and options.max_frames > 0:
        profile["max_frames"] = options.max_frames
        profile["min_frames"] = min(profile["min_frames"], options.max_frames)
    return profile


def compute_adaptive_frame_budget(
    frames: Sequence[FrameRecord],
    transcript_segments: Sequence[TranscriptSegment],
    *,
    options: PipelineOptions,
    profile: Dict[str, int],
) -> Optional[int]:
    max_frames = profile["max_frames"]
    if max_frames is None:
        return None

    if options.max_frames is not None and options.max_frames > 0:
        return min(options.max_frames, len(frames))

    min_frames = min(profile["min_frames"], max_frames)
    estimated_duration = estimate_duration_seconds(frames, transcript_segments)
    segment_count = len(transcript_segments)
    frame_count = len(frames)

    duration_factor = math.ceil(estimated_duration / 30.0) if estimated_duration > 0 else 1
    segment_factor = math.ceil(segment_count / 8.0) if segment_count > 0 else 1
    frame_factor = math.ceil(frame_count / 8.0) if frame_count > 0 else 1

    desired = max(min_frames, max(duration_factor, segment_factor, frame_factor) + 1)
    return min(max_frames, len(frames), desired)


def estimate_duration_seconds(
    frames: Sequence[FrameRecord],
    transcript_segments: Sequence[TranscriptSegment],
) -> float:
    duration = 0.0
    if transcript_segments:
        duration = max(duration, max(segment.end_seconds for segment in transcript_segments))
    if frames:
        duration = max(duration, max(frame.timestamp_seconds for frame in frames))
    return duration


def choose_representative_frames(
    step: MergedStep,
    frames: Sequence[FrameRecord],
    *,
    limit: int,
) -> List[FrameRecord]:
    ranked = sorted(
        frames,
        key=lambda frame: rank_frame_for_step(frame, step),
        reverse=True,
    )
    chosen: List[FrameRecord] = []
    for frame in ranked:
        if not frame.visual and not frame.ocr_text:
            continue
        if frame.visual and any(frame.visual == existing.visual for existing in chosen):
            if not frame.ocr_text:
                continue
        if frame.ocr_text and any(frame.ocr_text == existing.ocr_text for existing in chosen):
            if not frame.visual:
                continue
        chosen.append(frame)
        if len(chosen) >= limit:
            break
    return chosen


def rank_frame_for_step(frame: FrameRecord, step: MergedStep) -> float:
    midpoint = (step.start_seconds + step.end_seconds) / 2
    distance = abs(frame.timestamp_seconds - midpoint)
    span = max(step.end_seconds - step.start_seconds, 1.0)
    proximity_score = max(0.0, 1.0 - (distance / (span + 3.0)))
    signal_score = 0.0
    if frame.visual:
        signal_score += 2.0
    if frame.ocr_text:
        signal_score += min(frame.ocr_score / 120.0, 2.0)
    if frame.scene_score is not None:
        signal_score += min(frame.scene_score * 2.0, 1.0)
    return signal_score + proximity_score


def rank_candidate_frame(frame: FrameRecord, step: MergedStep, spoken: str) -> float:
    midpoint = (step.start_seconds + step.end_seconds) / 2
    distance = abs(frame.timestamp_seconds - midpoint)
    span = max(step.end_seconds - step.start_seconds, 1.0)
    proximity_score = max(0.0, 1.0 - (distance / (span + 2.0)))
    scene_score = frame.scene_score or 0.0
    scene_bonus = min(scene_score * 3.0, 1.5)
    keyword_bonus = 0.5 if has_visual_priority_keywords(spoken) else 0.0
    return proximity_score + scene_bonus + keyword_bonus


def rank_segment_importance(spoken: str) -> float:
    spoken = clean_text(spoken).lower()
    if not spoken:
        return 0.0
    words = spoken.split()
    length_score = min(len(words) / 12.0, 1.5)
    keyword_bonus = 1.0 if has_visual_priority_keywords(spoken) else 0.0
    numeric_bonus = 0.35 if any(char.isdigit() for char in spoken) else 0.0
    return length_score + keyword_bonus + numeric_bonus


def has_visual_priority_keywords(spoken: str) -> bool:
    spoken = clean_text(spoken).lower()
    keywords = (
        "step",
        "wire",
        "connect",
        "solder",
        "code",
        "screen",
        "show",
        "display",
        "remote",
        "board",
        "module",
        "sensor",
        "drone",
        "price",
        "hash",
        "wallet",
    )
    return any(keyword in spoken for keyword in keywords)


def fallback_frame_priority(frame: FrameRecord) -> float:
    scene_score = frame.scene_score or 0.0
    return scene_score + 0.001 * frame.timestamp_seconds


def iter_ocr_variants(image) -> Iterable:
    image_module, image_ops_module, image_filter_module = import_pil_modules()
    rgb = image.convert("RGB")
    width, height = rgb.size
    regions = [
        rgb,
        rgb.crop((0, 0, width, max(1, int(height * 0.35)))),
        rgb.crop((0, int(height * 0.58), width, height)),
        rgb.crop((0, int(height * 0.2), width, int(height * 0.8))),
    ]
    for region in regions:
        gray = image_ops_module.grayscale(region)
        upscaled = gray.resize((gray.width * 2, gray.height * 2), image_module.Resampling.LANCZOS)
        auto = image_ops_module.autocontrast(upscaled)
        sharp = auto.filter(image_filter_module.SHARPEN)
        thresholded = sharp.point(lambda px: 255 if px > 165 else 0)
        yield auto
        yield thresholded


def extract_ocr_from_variant(pytesseract, image) -> tuple[str, float]:
    data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT,
        config="--psm 6",
    )
    line_buckets: Dict[tuple, List[tuple[str, float]]] = {}
    word_count = len(data.get("text", []))
    for index in range(word_count):
        raw_text = data["text"][index]
        text = sanitize_ocr_token(raw_text)
        if not text:
            continue
        confidence = safe_float(data["conf"][index])
        if confidence < 45:
            continue
        line_key = (
            data["block_num"][index],
            data["par_num"][index],
            data["line_num"][index],
        )
        line_buckets.setdefault(line_key, []).append((text, confidence))

    line_entries: List[tuple[str, float]] = []
    for words in line_buckets.values():
        tokens = [text for text, _confidence in words]
        line = clean_text(" ".join(tokens))
        if not is_useful_ocr_line(line):
            continue
        confidence = sum(confidence for _text, confidence in words) / max(len(words), 1)
        bonus = min(len(line) / 40.0, 1.5)
        line_entries.append((line, confidence + bonus))

    line_entries.sort(key=lambda item: item[1], reverse=True)
    deduped_lines = dedupe_preserve_order(line for line, _score in line_entries[:4])
    if not deduped_lines:
        return "", 0.0

    score = sum(score for _line, score in line_entries[: min(3, len(line_entries))]) / min(
        3, len(line_entries)
    )
    return " | ".join(deduped_lines), score


def sanitize_ocr_token(value: str) -> str:
    value = clean_text(value)
    value = value.replace("|", "I")
    value = re.sub(r"[^\w\s:/#.+\-()%$,@]", "", value)
    return value.strip()


def is_useful_ocr_line(value: str) -> bool:
    if not value or len(value) < 3:
        return False
    letters = sum(char.isalpha() for char in value)
    digits = sum(char.isdigit() for char in value)
    spaces = value.count(" ")
    useful_chars = letters + digits
    if useful_chars < 3:
        return False
    if spaces == 0 and len(value) > 32:
        return False
    if digits == 0 and letters < 3:
        return False
    return True


def postprocess_visual_text(value: str) -> str:
    value = clean_text(value)
    value = re.sub(r"^the image shows\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^the frame shows\s+", "", value, flags=re.IGNORECASE)
    return value[:280]


def choose_on_screen_text(
    *,
    tesseract_text: str,
    tesseract_score: float,
    vision_text: str,
) -> str:
    tesseract_text = normalize_on_screen_text(tesseract_text)
    vision_text = normalize_on_screen_text(vision_text)

    if vision_text and not tesseract_text:
        return vision_text
    if tesseract_text and not vision_text:
        return tesseract_text
    if not tesseract_text and not vision_text:
        return ""

    tesseract_noise = estimate_text_noise(tesseract_text)
    vision_noise = estimate_text_noise(vision_text)

    if vision_text and (tesseract_score < 65 or tesseract_noise > vision_noise + 0.15):
        return vision_text
    if tesseract_text and vision_noise > tesseract_noise + 0.15:
        return tesseract_text
    return merge_text_signals(tesseract_text, vision_text)


def normalize_on_screen_text(value: str) -> str:
    value = clean_text(value)
    if not value:
        return ""
    parts = [clean_text(part) for part in value.split("|")]
    useful = [part for part in parts if is_useful_ocr_line(part)]
    return " | ".join(dedupe_preserve_order(useful))


def estimate_text_noise(value: str) -> float:
    if not value:
        return 1.0
    useful = sum(char.isalnum() or char in " .,:/#-+%$()" for char in value)
    return 1.0 - (useful / max(len(value), 1))


def merge_text_signals(primary: str, secondary: str) -> str:
    parts = []
    if primary:
        parts.extend(clean_text(part) for part in primary.split("|"))
    if secondary:
        parts.extend(clean_text(part) for part in secondary.split("|"))
    useful = [part for part in parts if is_useful_ocr_line(part)]
    return " | ".join(dedupe_preserve_order(useful[:4]))


def parse_vision_response(raw: str) -> Dict[str, str]:
    raw = clean_text(raw)
    parsed = try_parse_vision_json(raw)
    if parsed is not None:
        return parsed

    json_match = re.search(r"\{.*\}", raw)
    if json_match:
        parsed = try_parse_vision_json(json_match.group(0))
        if parsed is not None:
            return parsed

    return {
        "visual": postprocess_visual_text(raw),
        "on_screen_text": "",
    }


def try_parse_vision_json(raw: str) -> Optional[Dict[str, str]]:
    try:
        data = json.loads(raw)
    except Exception:
        return None
    return {
        "visual": postprocess_visual_text(str(data.get("visual", ""))),
        "on_screen_text": normalize_on_screen_text(str(data.get("on_screen_text", ""))),
    }


def import_pil_modules():
    try:
        from PIL import Image, ImageFilter, ImageOps
    except ImportError as exc:  # pragma: no cover - import guard
        raise Reel2MdError("Python package 'Pillow' is not installed.") from exc
    return Image, ImageOps, ImageFilter


def safe_float(value: object) -> float:
    try:
        if value is None:
            return -math.inf
        return float(value)
    except (TypeError, ValueError):
        return -math.inf
