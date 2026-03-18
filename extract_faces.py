"""
extract_faces.py
Extracts face crops from CCTV video(s) in Video_1/ folder.
Saves cropped faces to raw_faces/ folder for the enhancement pipeline.
"""

import cv2
import os
from pathlib import Path

VIDEO_DIR = Path("Video_1")
OUTPUT_DIR = Path("raw_faces")
OUTPUT_DIR.mkdir(exist_ok=True)

# Sample every N frames to avoid duplicates (CCTV has many similar frames)
FRAME_SAMPLE_INTERVAL = 15
# Minimum face size in pixels to save
MIN_FACE_SIZE = 12
# Maximum faces to extract (assignment says ~100)
MAX_FACES = 100
# Padding around detected face (fraction of face size)
# Minimal padding to keep crops small (50-80px) — the enhancement pipeline
# is designed for tiny 12-80px CCTV face crops
PADDING = 0.05

def extract_faces_from_video(video_path, start_count=0):
    """Extract face crops from a single video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return start_count

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Video: {video_path.name} | {total_frames} frames | {fps:.1f} FPS")

    # Use OpenCV's Haar cascade for face detection (no dlib needed)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    # Also try profile face detector for side views
    profile_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_profileface.xml"
    )

    count = start_count
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or count >= MAX_FACES:
            break

        if frame_idx % FRAME_SAMPLE_INTERVAL != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect frontal faces
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
        )

        # If no frontal faces, try profile
        if len(faces) == 0:
            faces = profile_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
            )

        h_frame, w_frame = frame.shape[:2]

        for (x, y, w, h) in faces:
            if count >= MAX_FACES:
                break

            # Add padding
            pad_x = int(w * PADDING)
            pad_y = int(h * PADDING)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w_frame, x + w + pad_x)
            y2 = min(h_frame, y + h + pad_y)

            face_crop = frame[y1:y2, x1:x2]

            if face_crop.shape[0] < MIN_FACE_SIZE or face_crop.shape[1] < MIN_FACE_SIZE:
                continue

            count += 1
            fname = f"face_{count:04d}.jpg"
            cv2.imwrite(str(OUTPUT_DIR / fname), face_crop)

            if count % 10 == 0:
                print(f"    Extracted {count} faces so far...")

        frame_idx += 1

    cap.release()
    return count


if __name__ == "__main__":
    video_files = sorted(VIDEO_DIR.glob("*"))
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}
    video_files = [v for v in video_files if v.suffix.lower() in video_exts]

    if not video_files:
        print("No video files found in Video_1/")
        exit(1)

    print(f"Found {len(video_files)} video(s)")
    total = 0

    for vf in video_files:
        total = extract_faces_from_video(vf, start_count=total)

    print(f"\nDone! Extracted {total} face crops into {OUTPUT_DIR}/")
    print(f"Now run: python solution.py")
