"""
DeepShield — Video Preprocessor
Extracts frames from video bytes using OpenCV.
"""

import cv2
import numpy as np
import tempfile
import os
from PIL import Image
from typing import List, Tuple


def extract_frames(
    video_bytes: bytes,
    num_frames: int = 16,
    target_size: tuple = (224, 224),
) -> Tuple[List[np.ndarray], float, int]:
    """
    Extract evenly-spaced frames from video bytes.
    Returns (frames_list, fps, total_frame_count)
    Each frame is numpy array (H, W, 3) uint8 RGB.
    """
    # Write to temp file
    suffix = ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

        if total_frames <= 0:
            cap.release()
            return [], fps, 0

        # Choose evenly-spaced frame indices
        indices = np.linspace(0, total_frames - 1, num=min(num_frames, total_frames), dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, target_size)
                frames.append(frame_resized)

        cap.release()
        return frames, fps, total_frames

    finally:
        os.unlink(tmp_path)


def frames_to_pil(frames: List[np.ndarray]) -> List[Image.Image]:
    return [Image.fromarray(f) for f in frames]
