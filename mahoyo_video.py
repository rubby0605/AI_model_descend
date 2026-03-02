#!/usr/bin/env python3
"""
魔法使之夜 第13章「星之語」— AI 影片生成腳本
使用 Google Gemini API + Veo 3.1 生成各場景影片，再用 ffmpeg 串接

Usage:
    python mahoyo_video.py              # 生成全部場景
    python mahoyo_video.py --scene 3    # 只生成第 3 個場景
    python mahoyo_video.py --concat     # 只做串接（場景已生成完畢時）
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from google import genai
from google.genai import types

# ── Config ──
API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = "veo-3.1-generate-preview"
OUTPUT_DIR = Path(__file__).parent / "mahoyo_ch13_clips"
FINAL_OUTPUT = Path(__file__).parent / "mahoyo_ch13_star_language.mp4"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"

# Veo 3.1 settings
VIDEO_DURATION = 8        # seconds per clip
ASPECT_RATIO = "16:9"
RESOLUTION = "720p"       # 720p for faster generation; change to 1080p if needed

# ── Visual style prefix (appended to every prompt) ──
STYLE = (
    "Anime visual novel cinematic style, inspired by ufotable and TYPE-MOON. "
    "Soft moonlight, winter atmosphere, breath visible in cold air. "
    "Beautiful detailed backgrounds with painterly lighting. "
    "Characters drawn in Japanese anime style with expressive eyes. "
    "Cinematic composition, shallow depth of field, film grain. "
)

# ══════════════════════════════════════════════════════════════
# Scene definitions — 第13章「星之語」分鏡
# ══════════════════════════════════════════════════════════════
SCENES = [
    {
        "id": 1,
        "title": "客廳 — 青子宣布出發",
        "prompt": (
            "Interior of a Western-style mansion living room at night, warm amber lighting. "
            "A confident red-haired girl (Aoko, age 17, school uniform with red ribbon) stands "
            "at the doorway announcing something decisive. A brown-haired boy (Soujuurou, plain clothes) "
            "sits on the sofa looking confused. A quiet dark-haired girl (Aizu) sits opposite, "
            "teacup in hand, eyes widening in surprise. Japanese anime style."
        ),
    },
    {
        "id": 2,
        "title": "秋古城車站 — 深夜的小站",
        "prompt": (
            "A tiny rural Japanese train station at night, nearly 11 PM. Only a single warm light "
            "illuminates the platform. An elderly station attendant dozes beside a small heater and TV. "
            "Two figures — a red-haired girl and a brown-haired boy in winter coats — step off an empty "
            "train onto the deserted platform. Vast dark countryside surrounds the station like a lone "
            "star in a dark ocean. Winter cold, visible breath. Anime cinematic."
        ),
    },
    {
        "id": 3,
        "title": "田間小路 — 月光下的夜行",
        "prompt": (
            "Wide shot of two silhouetted figures walking along a narrow dirt path between dormant "
            "winter wheat fields at night. Brilliant full moon overhead casting silver light. "
            "Occasional power line poles with dim orange lamps create pools of light. "
            "The girl walks with hands in coat pockets, the boy follows beside her. "
            "Vast dark plains stretching to a distant mountain with a single house light. "
            "Peaceful winter atmosphere, anime art style, cinematic wide angle."
        ),
    },
    {
        "id": 4,
        "title": "山道對話 — 根源之渦",
        "prompt": (
            "Close-up two-shot of a red-haired girl and a brown-haired boy walking on a dark "
            "tree-lined mountain path. Moonlight filters through bare winter branches creating "
            "dappled patterns. The girl gestures while explaining something passionately, white breath "
            "visible. The boy listens intently with a slight head tilt. Anime character close-up, "
            "warm expressions, winter night atmosphere, visible constellation overhead."
        ),
    },
    {
        "id": 5,
        "title": "記憶消除宣告 — 青子停步",
        "prompt": (
            "Dramatic moment on a dark mountain road. A red-haired girl suddenly stops walking, "
            "turning to face a brown-haired boy with a serious, pained expression. Moonlight "
            "illuminates her face. The boy also stops, his expression shifting from confusion to "
            "quiet acceptance. A warm house light visible far behind them on the hillside. "
            "Emotional anime scene, dramatic lighting contrast, winter night."
        ),
    },
    {
        "id": 6,
        "title": "草十郎的自白 — 山中往事",
        "prompt": (
            "A brown-haired boy walks forward on a mountain path, speaking to the night sky "
            "rather than to the girl beside him. His expression is distant, nostalgic. "
            "Brief overlay/montage feel: misty mountain wilderness, a solitary child among trees, "
            "wolves in forest shadows. The red-haired girl walks silently beside him, listening "
            "with an unexpectedly gentle expression. Stars visible through bare branches above. "
            "Melancholic anime atmosphere, soft focus."
        ),
    },
    {
        "id": 7,
        "title": "離別 — 森林入口",
        "prompt": (
            "A warm traditional Japanese house at the end of a mountain road, light glowing from "
            "windows. A red-haired girl points toward a dark forest path behind the house. "
            "A brown-haired boy raises one hand in farewell without looking back as he walks "
            "toward the dark forest entrance. The girl watches his retreating figure with a sad "
            "half-smile. Emotional farewell scene, anime cinematic, moonlit winter night."
        ),
    },
    {
        "id": 8,
        "title": "洞穴中的祖父 — 神秘對話",
        "prompt": (
            "Interior of a mystical cave glowing with ethereal blue-white mist. A translucent, "
            "smoke-like figure of an ancient man hovers in the center — more spirit than flesh, "
            "features impossible to discern. A brown-haired boy stands before this apparition, "
            "looking up with quiet determination. Magical particles float in the air. "
            "TYPE-MOON magical aesthetic, otherworldly atmosphere, anime fantasy scene."
        ),
    },
    {
        "id": 9,
        "title": "青子回家 — 我回來了",
        "prompt": (
            "A red-haired girl stands at the front door of a warm country house, hand on the "
            "doorknob. She takes a deep breath, then opens the door. Warm golden light spills "
            "out from inside. Her expression softens into a genuine, gentle smile as tears form "
            "at the corners of her eyes. She mouths the words 'I'm home.' "
            "Emotional anime scene, warm interior light vs cold dark exterior, intimate close-up."
        ),
    },
    {
        "id": 10,
        "title": "意外重逢 — 草十郎還在",
        "prompt": (
            "Outside a country house, a red-haired girl bursts out the front door to find a "
            "brown-haired boy standing there gazing at the starry sky with a peaceful smile. "
            "She rushes toward him, expression shifting rapidly between anger, disbelief, and joy. "
            "He looks at her with innocent confusion. Starry winter sky background. "
            "Comedy-drama anime moment, expressive character acting, dynamic poses."
        ),
    },
    {
        "id": 11,
        "title": "歸途 — 星空下的對話",
        "prompt": (
            "Two figures walking side by side on a dark country road back toward a distant glowing "
            "train station. The boy looks up at the magnificent star-filled sky with mixed emotions — "
            "longing and acceptance. The girl walks beside him, also gazing upward. "
            "Their white breath mingles in the cold air. Spectacular anime starfield, "
            "Milky Way visible, deeply emotional and beautiful night scene."
        ),
    },
    {
        "id": 12,
        "title": "新年鐘聲 — 新年快樂",
        "prompt": (
            "The boy suddenly turns to the red-haired girl with a bright, genuine smile and says "
            "'Happy New Year.' The girl freezes, eyes widening in surprise, then slowly breaks into "
            "a warm, beautiful smile. Distant temple bells ring. The camera slowly pulls back to "
            "reveal the two small figures under a vast canopy of stars on an empty country road. "
            "A tiny train station glows like a lighthouse in the distance. "
            "Breathtaking anime finale, emotional climax, starlit winter night, tears of joy."
        ),
    },
]


def load_progress():
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {}


def save_progress(progress):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def generate_scene(client, scene, progress):
    scene_id = str(scene["id"])
    output_file = OUTPUT_DIR / f"scene_{scene['id']:02d}.mp4"

    # Skip if already generated
    if scene_id in progress and output_file.exists():
        print(f"  ✓ Scene {scene['id']} already generated, skipping")
        return output_file

    full_prompt = STYLE + scene["prompt"]

    print(f"\n{'='*60}")
    print(f"  Scene {scene['id']}: {scene['title']}")
    print(f"{'='*60}")
    print(f"  Generating {VIDEO_DURATION}s video at {RESOLUTION}...")

    try:
        operation = client.models.generate_videos(
            model=MODEL,
            prompt=full_prompt,
            config=types.GenerateVideosConfig(
                aspect_ratio=ASPECT_RATIO,
                duration_seconds=VIDEO_DURATION,
            ),
        )

        # Poll until done
        poll_count = 0
        while not operation.done:
            poll_count += 1
            elapsed = poll_count * 15
            print(f"  Waiting... ({elapsed}s elapsed)", end="\r")
            time.sleep(15)
            operation = client.operations.get(operation)

        print(f"  Generation complete!                    ")

        # Download and save
        generated_video = operation.response.generated_videos[0]
        client.files.download(file=generated_video.video)
        generated_video.video.save(str(output_file))

        print(f"  Saved: {output_file}")

        # Update progress
        progress[scene_id] = {
            "title": scene["title"],
            "file": str(output_file),
            "status": "done",
        }
        save_progress(progress)

        return output_file

    except Exception as e:
        print(f"  ERROR generating scene {scene['id']}: {e}")
        progress[scene_id] = {"title": scene["title"], "status": f"error: {e}"}
        save_progress(progress)
        return None


def concatenate_videos(clip_files):
    """Use ffmpeg to concatenate all scene clips into one video."""
    import subprocess

    # Create concat file list
    concat_list = OUTPUT_DIR / "concat_list.txt"
    with open(concat_list, "w") as f:
        for clip in clip_files:
            f.write(f"file '{clip}'\n")

    print(f"\nConcatenating {len(clip_files)} clips into final video...")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        str(FINAL_OUTPUT),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Final video saved: {FINAL_OUTPUT}")
    else:
        # Try re-encoding if concat copy fails (different codecs)
        print("  Direct concat failed, re-encoding...")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list),
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "medium",
            "-c:a", "aac",
            "-b:a", "192k",
            str(FINAL_OUTPUT),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Final video saved: {FINAL_OUTPUT}")
        else:
            print(f"  FFmpeg error: {result.stderr[:500]}")


def main():
    parser = argparse.ArgumentParser(description="魔法使之夜 Ch.13 Video Generator")
    parser.add_argument("--scene", type=int, help="Generate only this scene number (1-12)")
    parser.add_argument("--concat", action="store_true", help="Only concatenate existing clips")
    parser.add_argument("--list", action="store_true", help="List all scenes")
    parser.add_argument("--duration", type=int, default=8, choices=[4, 6, 8],
                        help="Video duration per scene (default: 8)")
    args = parser.parse_args()

    global VIDEO_DURATION
    VIDEO_DURATION = args.duration

    # List scenes
    if args.list:
        print("\n魔法使之夜 第13章「星之語」— 場景列表\n")
        for s in SCENES:
            print(f"  Scene {s['id']:2d}: {s['title']}")
        print(f"\n  Total: {len(SCENES)} scenes × {VIDEO_DURATION}s = {len(SCENES) * VIDEO_DURATION}s")
        return

    # Setup
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Concat only
    if args.concat:
        clip_files = sorted(OUTPUT_DIR.glob("scene_*.mp4"))
        if not clip_files:
            print("No clips found to concatenate!")
            return
        concatenate_videos(clip_files)
        return

    # Initialize client
    if not API_KEY:
        print("ERROR: Set GEMINI_API_KEY environment variable")
        sys.exit(1)

    client = genai.Client(api_key=API_KEY)
    progress = load_progress()

    # Filter scenes
    if args.scene:
        scenes_to_gen = [s for s in SCENES if s["id"] == args.scene]
        if not scenes_to_gen:
            print(f"Scene {args.scene} not found (valid: 1-{len(SCENES)})")
            return
    else:
        scenes_to_gen = SCENES

    print(f"\n{'#'*60}")
    print(f"  魔法使之夜 第13章「星之語」— Video Generation")
    print(f"  Scenes: {len(scenes_to_gen)} | Duration: {VIDEO_DURATION}s each")
    print(f"  Model: {MODEL} | Resolution: {RESOLUTION}")
    print(f"  Output: {OUTPUT_DIR}/")
    print(f"{'#'*60}")

    # Generate scenes
    generated_files = []
    for scene in scenes_to_gen:
        result = generate_scene(client, scene, progress)
        if result:
            generated_files.append(result)

        # Rate limiting — be nice to the API
        if scene != scenes_to_gen[-1]:
            print("  Cooling down 5s before next scene...")
            time.sleep(5)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Generation Summary")
    print(f"{'='*60}")
    success = len(generated_files)
    total = len(scenes_to_gen)
    print(f"  Success: {success}/{total}")

    if success < total:
        failed = [s["id"] for s in scenes_to_gen
                  if not (OUTPUT_DIR / f"scene_{s['id']:02d}.mp4").exists()]
        print(f"  Failed: {failed}")
        print(f"  Re-run with --scene N to retry individual scenes")

    # Concatenate if all scenes were generated
    if not args.scene:
        all_clips = sorted(OUTPUT_DIR.glob("scene_*.mp4"))
        if len(all_clips) == len(SCENES):
            concatenate_videos(all_clips)
        else:
            print(f"\n  {len(all_clips)}/{len(SCENES)} clips ready. "
                  f"Run with --concat after all scenes are generated.")


if __name__ == "__main__":
    main()
