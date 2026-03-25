"""
Video Utilities Module for PRISM
================================
Core video processing utilities including encoding, cropping, and face detection.

Features:
- GPU-accelerated encoding (NVENC when available)
- Smart cropping with face detection
- Video file hashing for caching
- Transcript word filtering optimization
"""

import os
import hashlib
import subprocess
from typing import List, Dict, Any, Optional, Tuple
from moviepy import VideoFileClip

# Global caches
_face_cascade: Any = None
_gpu_encoder: Optional[str] = None

# Default dimensions
DEFAULT_WIDTH = 1080
DEFAULT_HEIGHT = 1920


def get_video_codec() -> str:
    """Detect best available video codec. Prefers NVENC GPU encoding."""
    global _gpu_encoder
    if _gpu_encoder is not None:
        return _gpu_encoder
    
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=5
        )
        if "h264_nvenc" in result.stdout:
            _gpu_encoder = "h264_nvenc"
            print("[Performance] Using NVIDIA NVENC GPU encoder")
            return _gpu_encoder
    except Exception as e:
        print(f"[Performance] Could not detect GPU encoder: {e}")
    
    _gpu_encoder = "libx264"
    print("[Performance] Using libx264 CPU encoder")
    return _gpu_encoder


def get_ffmpeg_params(codec: str) -> Tuple[str, List[str]]:
    """Get optimal FFmpeg parameters based on codec."""
    if codec == "h264_nvenc":
        # NVENC settings: -cq is quality (0-51, lower=better), p4 is balanced preset
        return ("p4", ["-cq", "23", "-b:v", "0"])
    else:
        # libx264 settings: -crf is quality (18-28), fast preset
        return ("fast", ["-crf", "23"])


def get_face_cascade():
    """Lazy-load and cache the Haar cascade classifier (expensive to load)."""
    global _face_cascade
    if _face_cascade is None:
        try:
            import cv2
            cascade_path: str = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'  # type: ignore[attr-defined]
            _face_cascade = cv2.CascadeClassifier(cascade_path)
            print("[Performance] Face cascade loaded and cached.")
        except Exception as e:
            print(f"[Performance] Could not load face cascade: {e}")
            _face_cascade = False  # Mark as failed so we don't retry
    return _face_cascade if _face_cascade else None


def detect_face_offset(clip: VideoFileClip, new_w: int, src_w: int) -> Optional[int]:
    """Detect face position and return x_offset for cropping. Returns None if no face found."""
    try:
        import cv2
        face_cascade = get_face_cascade()
        if face_cascade is None:
            return None
            
        # Sample a frame from the middle of the clip
        sample_time = clip.duration / 2
        frame = clip.get_frame(sample_time)
        
        # Convert RGB to Grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            # Find the largest face by area
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            face_center_x = int(x + w / 2)
            x_offset = int(max(0, min(src_w - new_w, face_center_x - (new_w // 2))))
            print(f"[Smart Crop] Found face at x={face_center_x}, crop offset={x_offset}")
            return x_offset
    except Exception as e:
        print(f"[Smart Crop] Face detection error: {e}")
    return None


def crop_to_vertical(
    clip: VideoFileClip,
    start_time: Optional[float] = None,
    face_x_offset: Optional[int] = None,
    target_w: int = DEFAULT_WIDTH,
    target_h: int = DEFAULT_HEIGHT
) -> VideoFileClip:
    """Smart center-crop a clip to 9:16 aspect ratio. Optionally uses pre-computed face offset."""
    src_w, src_h = clip.w, clip.h
    src_aspect = src_w / src_h
    target_aspect = target_w / target_h

    if src_aspect > target_aspect:
        # Source is wider than 9:16 — crop width
        new_w = int(src_h * target_aspect)
        
        # Use provided face offset, or fall back to center crop
        if face_x_offset is not None:
            x_offset = face_x_offset
        else:
            x_offset = (src_w - new_w) // 2

        clip = clip.cropped(x1=x_offset, y1=0, x2=x_offset + new_w, y2=src_h)
    elif src_aspect < target_aspect:
        # Source is taller than 9:16 — crop height
        new_h = int(src_w / target_aspect)
        y_offset = (src_h - new_h) // 2
        clip = clip.cropped(x1=0, y1=y_offset, x2=src_w, y2=y_offset + new_h)

    # Resize to exact target resolution
    clip = clip.resized((target_w, target_h))
    return clip


def get_relevant_words_for_cuts(
    transcript_segments: List[Dict[str, Any]],
    cuts: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Pre-filter transcript words to only those overlapping with cuts.
    This avoids iterating the full transcript for every cut.
    Returns a flat list of {word, start, end} dicts sorted by start time.
    """
    # Build time ranges from cuts
    cut_ranges = [(max(0, c["start"]), c["end"]) for c in cuts]
    
    relevant_words = []
    for seg in transcript_segments:
        words_data = seg.get("words", [])
        if not words_data:
            words_data = [{"word": seg["text"], "start": seg["start"], "end": seg["end"]}]
        
        for word_info in words_data:
            word_start = word_info["start"]
            word_end = word_info["end"]
            word_text = word_info.get("word", "").strip()
            
            if not word_text:
                continue
                
            # Check if word overlaps with ANY cut
            for cut_start, cut_end in cut_ranges:
                if word_start < cut_end and word_end > cut_start:
                    relevant_words.append({
                        "word": word_text,
                        "start": word_start,
                        "end": word_end
                    })
                    break  # Don't add same word multiple times
    
    return sorted(relevant_words, key=lambda w: w["start"])


def get_video_hash(video_path: str) -> str:
    """Calculate MD5 hash of video file for cache identification."""
    hash_md5 = hashlib.md5()
    with open(video_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()[:16]