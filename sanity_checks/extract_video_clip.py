import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import PROJECT_ROOT, VIDEOS_ROOT


def extract_clip_by_frames(
    video_path: str, output_path: str, start_frame: int, end_frame: int
):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_frame < 0 or start_frame >= total_frames:
        raise ValueError(
            f"Invalid start_frame {start_frame}. Total frames: {total_frames}"
        )
    if end_frame < start_frame:
        raise ValueError(
            f"end_frame ({end_frame}) cannot be less than start_frame ({start_frame})"
        )

    end_frame = min(end_frame, total_frames - 1)

    start_time = start_frame / fps
    end_time = end_frame / fps

    print(
        f"Extracting frames {start_frame} to {end_frame} "
        f"(Time: {start_time:.2f}s to {end_time:.2f}s) from {video_path}"
    )
    print(f"Video properties: {width}x{height} @ {fps} fps")

    import subprocess

    # Define the codec and create VideoWriter object
    # We use 'mp4v' first, then convert it using ffmpeg to avoid OpenCV codec initialization issues
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    temp_output_path = output_path.replace(".mp4", "_temp.mp4")

    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

    # Seek to the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame = start_frame
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            print(
                f"Warning: Could not read frame {current_frame}. Video might have ended prematurely."
            )
            break

        out.write(frame)
        current_frame += 1

    # Release everything if job is finished
    cap.release()
    out.release()

    print("Converting to H.264 for better compatibility... Please wait.")
    try:
        # Use python's subprocess to gracefully run ffmpeg and convert to VSCode-compatible H.264
        subprocess.run(
            ["ffmpeg", "-y", "-i", temp_output_path, "-vcodec", "libx264", output_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        print(f"Successfully saved extracted clip to: {output_path}")
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"Warning: ffmpeg conversion failed ({e}). Keeping original format.")
        os.rename(temp_output_path, output_path)
        print(f"Successfully saved extracted clip to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract a frame range from a source video for table consistency checks."
    )
    parser.add_argument("--video_name", type=str, required=True, help="Video basename without .mp4")
    parser.add_argument("--start_frame", type=int, required=True)
    parser.add_argument("--end_frame", type=int, required=True)
    parser.add_argument(
        "--videos_root",
        type=str,
        default=VIDEOS_ROOT,
        help="Directory containing merged source videos.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output mp4 path. Default: extracted_clip/<video>_<start>_<end>.mp4",
    )
    args = parser.parse_args()

    video_path = os.path.join(args.videos_root, f"{args.video_name}.mp4")
    output_path = args.output_path or os.path.join(
        PROJECT_ROOT,
        "extracted_clip",
        f"{args.video_name}_{args.start_frame}_{args.end_frame}.mp4",
    )

    extract_clip_by_frames(
        video_path=video_path,
        output_path=output_path,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )


if __name__ == "__main__":
    main()
