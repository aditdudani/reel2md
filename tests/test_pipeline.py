import unittest
from pathlib import Path

from reel2md.pipeline import (
    compute_adaptive_frame_budget,
    estimate_duration_seconds,
    FrameRecord,
    MediaInfo,
    MergedStep,
    PipelineOptions,
    SPEED_PROFILES,
    TranscriptSegment,
    build_output_path,
    choose_on_screen_text,
    choose_representative_frames,
    choose_title,
    format_timestamp,
    load_output_index,
    matches_index_path,
    merge_modalities,
    parse_ffmpeg_metadata_output,
    parse_showinfo_output,
    parse_vision_response,
    resolve_speed_profile,
    render_markdown,
    select_frames_for_analysis,
    should_fallback_scene_extraction,
    update_output_index,
)


class PipelineTests(unittest.TestCase):
    def test_parse_ffmpeg_metadata_output(self):
        text = "\n".join(
            [
                "frame 0 pts 123 pts_time 1.230000",
                "lavfi.scene_score=0.412300",
                "frame 1 pts 456 pts_time 4.560000",
                "lavfi.scene_score=0.500000",
            ]
        )
        parsed = parse_ffmpeg_metadata_output(text)
        self.assertEqual(2, len(parsed))
        self.assertEqual(1.23, parsed[0]["pts_time"])
        self.assertEqual(0.5, parsed[1]["scene_score"])

    def test_merge_modalities_assigns_frames_to_segments(self):
        segments = [
            TranscriptSegment(start_seconds=0.0, end_seconds=5.0, spoken="Intro"),
            TranscriptSegment(start_seconds=5.0, end_seconds=10.0, spoken="Build"),
        ]
        frames = [
            FrameRecord(timestamp_seconds=1.0, image_path=Path("frame1.jpg"), visual="Board", ocr_text="GPIO 34"),
            FrameRecord(timestamp_seconds=7.5, image_path=Path("frame2.jpg"), visual="Sensor", ocr_text="AOUT"),
        ]
        steps = merge_modalities(segments, frames)
        self.assertEqual(2, len(steps))
        self.assertEqual(["Board"], steps[0].visuals)
        self.assertEqual(["AOUT"], steps[1].on_screen_text)

    def test_render_markdown_contains_expected_sections(self):
        media = MediaInfo(
            source_url="https://example.com/reel",
            canonical_url="https://example.com/reel",
            platform="instagram",
            video_id="abc123",
            title="Test Reel",
            uploader="tester",
            description="desc",
            duration_seconds=12,
        )
        options = PipelineOptions(
            output_dir=None,
            vision_model="llava",
            whisper_model="base",
            ollama_host="http://127.0.0.1:11434",
            scene_threshold=0.35,
            frame_workers=1,
        )
        steps = [
            MergedStep(
                start_seconds=0,
                end_seconds=5,
                spoken="Connect power.",
                visuals=["ESP32 board on desk."],
                on_screen_text=["3V3 | GND"],
            )
        ]
        markdown = render_markdown(media=media, steps=steps, options=options)
        self.assertIn("## Steps", markdown)
        self.assertIn("- Spoken: Connect power.", markdown)
        self.assertIn("- Visual: ESP32 board on desk.", markdown)
        self.assertIn("- On-screen text: 3V3 | GND", markdown)

    def test_format_timestamp_supports_hours(self):
        self.assertEqual("01:01:01", format_timestamp(3661))

    def test_parse_showinfo_output(self):
        text = "\n".join(
            [
                "[Parsed_showinfo_1 @ 000002] n:   0 pts:   1536 pts_time:0.100000 pos: 123 fmt:yuv420p",
                "[Parsed_showinfo_1 @ 000002] n:   1 pts:  46080 pts_time:3.000000 pos: 456 fmt:yuv420p",
            ]
        )
        parsed = parse_showinfo_output(text)
        self.assertEqual(2, len(parsed))
        self.assertEqual(0.1, parsed[0]["pts_time"])
        self.assertEqual(3.0, parsed[1]["pts_time"])

    def test_should_fallback_scene_extraction_on_empty_scene_pass(self):
        stderr_text = "\n".join(
            [
                "No filtered frames for output stream, trying to initialize anyway.",
                "Nothing was written into output file, because at least one of its streams received no packets.",
                "Conversion failed!",
            ]
        )
        self.assertTrue(should_fallback_scene_extraction(stderr_text, [], []))

    def test_build_output_path_uses_title(self):
        media = MediaInfo(
            source_url="https://example.com/reel",
            canonical_url="https://example.com/reel",
            platform="instagram",
            video_id="abc123",
            title="How To Build A Drone",
            uploader="tester",
            description="desc",
            duration_seconds=12,
            dedupe_key="instagram:abc123",
        )
        path = build_output_path(Path("output"), media, index={})
        self.assertEqual("how-to-build-a-drone.md", path.name)

    def test_choose_title_prefers_description_when_title_is_generic(self):
        title = choose_title(
            raw_title="Video by patsunrick",
            description="How to control a drone with your hand. Comment drone for the full guide.",
            uploader="Pat",
            platform="instagram",
            video_id="abc123",
        )
        self.assertEqual("How to control a drone with your hand", title)

    def test_choose_representative_frames_limits_count(self):
        step = MergedStep(
            start_seconds=10,
            end_seconds=20,
            spoken="Build step",
            visuals=[],
            on_screen_text=[],
        )
        frames = [
            FrameRecord(timestamp_seconds=12, image_path=Path("a.jpg"), visual="A", ocr_text="", ocr_score=0),
            FrameRecord(timestamp_seconds=13, image_path=Path("b.jpg"), visual="B", ocr_text="GPIO 1", ocr_score=80),
            FrameRecord(timestamp_seconds=14, image_path=Path("c.jpg"), visual="C", ocr_text="GPIO 2", ocr_score=70),
        ]
        chosen = choose_representative_frames(step, frames, limit=2)
        self.assertEqual(2, len(chosen))

    def test_choose_on_screen_text_prefers_cleaner_vision_text(self):
        text = choose_on_screen_text(
            tesseract_text="i th 1 | 2 y a -",
            tesseract_score=50,
            vision_text="Step 1 | Buy toy drone on Amazon",
        )
        self.assertEqual("Step 1 | Buy toy drone on Amazon", text)

    def test_parse_vision_response_json(self):
        parsed = parse_vision_response(
            '{"visual":"A person soldering wires.","on_screen_text":"Step 2 | Wire the Raspberry Pi"}'
        )
        self.assertEqual("A person soldering wires.", parsed["visual"])
        self.assertEqual("Step 2 | Wire the Raspberry Pi", parsed["on_screen_text"])

    def test_output_index_round_trip(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            media = MediaInfo(
                source_url="https://example.com/reel",
                canonical_url="https://example.com/reel",
                platform="instagram",
                video_id="abc123",
                title="How To Build A Drone",
                uploader="tester",
                description="desc",
                duration_seconds=12,
                dedupe_key="instagram:abc123",
            )
            output_path = output_dir / "how-to-build-a-drone.md"
            output_path.write_text("test", encoding="utf-8")
            index = {}
            update_output_index(output_dir, index=index, media=media, output_path=output_path)
            reloaded = load_output_index(output_dir)
            self.assertTrue(matches_index_path(reloaded, media, output_path))

    def test_resolve_speed_profile_honors_override(self):
        options = PipelineOptions(
            output_dir=None,
            vision_model="llava",
            whisper_model="base",
            ollama_host="http://127.0.0.1:11434",
            scene_threshold=0.35,
            frame_workers=1,
            speed_mode="fast",
            max_frames=4,
        )
        profile = resolve_speed_profile(options)
        self.assertEqual(4, profile["max_frames"])
        self.assertEqual(SPEED_PROFILES["fast"]["max_frames_per_segment"], profile["max_frames_per_segment"])

    def test_compute_adaptive_frame_budget_uses_less_than_fast_cap_for_simple_input(self):
        options = PipelineOptions(
            output_dir=None,
            vision_model="llava",
            whisper_model="base",
            ollama_host="http://127.0.0.1:11434",
            scene_threshold=0.35,
            frame_workers=1,
            speed_mode="fast",
        )
        frames = [
            FrameRecord(timestamp_seconds=float(index * 4), image_path=Path(f"{index}.jpg"), scene_score=0.5)
            for index in range(8)
        ]
        segments = [
            TranscriptSegment(start_seconds=0, end_seconds=4, spoken="Intro"),
            TranscriptSegment(start_seconds=4, end_seconds=8, spoken="Show result"),
            TranscriptSegment(start_seconds=8, end_seconds=12, spoken="Wrap up"),
        ]
        profile = resolve_speed_profile(options)
        budget = compute_adaptive_frame_budget(frames, segments, options=options, profile=profile)
        self.assertLess(budget, SPEED_PROFILES["fast"]["max_frames"])
        self.assertGreaterEqual(budget, SPEED_PROFILES["fast"]["min_frames"])

    def test_select_frames_for_analysis_caps_dense_reels(self):
        options = PipelineOptions(
            output_dir=None,
            vision_model="llava",
            whisper_model="base",
            ollama_host="http://127.0.0.1:11434",
            scene_threshold=0.35,
            frame_workers=1,
            speed_mode="fast",
        )
        segments = [
            TranscriptSegment(start_seconds=0, end_seconds=5, spoken="Step 1 connect wires"),
            TranscriptSegment(start_seconds=5, end_seconds=10, spoken="Step 2 write code"),
            TranscriptSegment(start_seconds=10, end_seconds=15, spoken="Step 3 test output"),
        ]
        frames = [
            FrameRecord(timestamp_seconds=float(index), image_path=Path(f"{index}.jpg"), scene_score=0.9 - (index * 0.01))
            for index in range(15)
        ]
        selected = select_frames_for_analysis(frames, segments, options=options)
        self.assertLessEqual(len(selected), 6)
        self.assertEqual(selected, sorted(selected, key=lambda frame: frame.timestamp_seconds))

    def test_estimate_duration_seconds_prefers_longer_transcript(self):
        frames = [FrameRecord(timestamp_seconds=10.0, image_path=Path("a.jpg"))]
        segments = [TranscriptSegment(start_seconds=0, end_seconds=18, spoken="Longer transcript")]
        self.assertEqual(18, estimate_duration_seconds(frames, segments))


if __name__ == "__main__":
    unittest.main()
