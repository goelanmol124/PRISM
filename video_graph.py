import os
import sys
import platform
import json
import warnings
import hashlib
import subprocess
from typing import TypedDict, List, Any, Dict, Optional, Union, cast
import datetime
import uuid

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from moviepy import VideoFileClip, concatenate_videoclips, ColorClip
import whisper
import torch
from llm_core import call_llm_with_structure, AnalysisResult, HeadingResult, CaptionAnalysis, CaptionGroup, CriticResult
from model_factory import ModelFactory
from critic_agent import CriticAgent

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Check for API Key
if not os.getenv("OPENROUTER_API_KEY"):
    raise ValueError("OPENROUTER_API_KEY not found in .env file")

# Dev mode cache directory
CACHE_DIR = ".dev_cache"

# --- Output Constants ---
TARGET_WIDTH = 512
TARGET_HEIGHT = 720
TARGET_ASPECT = TARGET_WIDTH / TARGET_HEIGHT  # 9:16 = 0.5625

# --- Performance: Cached OpenCV Face Cascade ---
_face_cascade: Any = None

# --- Performance: GPU Encoder Detection (cached) ---
_gpu_encoder: Optional[str] = None

def _get_video_codec() -> str:
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


def _get_ffmpeg_params(codec: str) -> tuple:
    """Get optimal FFmpeg parameters based on codec."""
    if codec == "h264_nvenc":
        # NVENC settings: -cq is quality (0-51, lower=better), p4 is balanced preset
        return ("p4", ["-cq", "23", "-b:v", "0"])
    else:
        # libx264 settings: -crf is quality (18-28), fast preset
        return ("fast", ["-crf", "23"])

def _get_face_cascade() -> Any:
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

# --- Performance: Pillow-based Text Rendering (replaces slow TextClip/ImageMagick) ---
_pillow_font_cache = {}

def _get_pillow_font(font_path: Optional[str], font_size: int):
    """Cache Pillow fonts to avoid repeated disk reads."""
    from PIL import ImageFont
    cache_key = (font_path, font_size)
    if cache_key not in _pillow_font_cache:
        try:
            if font_path:
                _pillow_font_cache[cache_key] = ImageFont.truetype(font_path, font_size)
            else:
                _pillow_font_cache[cache_key] = ImageFont.load_default()
        except Exception:
            _pillow_font_cache[cache_key] = ImageFont.load_default()
    return _pillow_font_cache[cache_key]


def _render_text_to_array(
    text: str,
    font_path: Optional[str],
    font_size: int,
    text_color: tuple = (255, 255, 0),      # Yellow (RGB)
    stroke_color: tuple = (0, 0, 0),         # Black
    stroke_width: int = 3,
    max_width: Optional[int] = None,
    padding: int = 10
):
    """
    Render text to a numpy array with transparency using Pillow.
    MUCH faster than TextClip which spawns ImageMagick subprocess.
    
    Returns: numpy array (H, W, 4) with RGBA channels
    """
    from PIL import Image, ImageDraw
    import numpy as np
    
    font = _get_pillow_font(font_path, font_size)
    
    # Create a temporary image to measure text size
    dummy_img = Image.new('RGBA', (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    
    # Get text bounding box
    bbox = dummy_draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    text_width = bbox[2] - bbox[0] + padding * 2
    text_height = bbox[3] - bbox[1] + padding * 2
    
    # Clamp to max width if specified
    if max_width and text_width > max_width:
        text_width = max_width
    
    # Create transparent image
    img = Image.new('RGBA', (int(text_width), int(text_height)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Center text in the image
    x = (text_width - (bbox[2] - bbox[0])) // 2
    y = (text_height - (bbox[3] - bbox[1])) // 2
    
    # Draw text with stroke (outline) then fill
    draw.text(
        (x, y), 
        text, 
        font=font, 
        fill=(*text_color, 255),
        stroke_width=stroke_width,
        stroke_fill=(*stroke_color, 255)
    )
    
    return np.array(img)


def _create_text_image_clip(
    text: str,
    font_path: Optional[str],
    font_size: int,
    text_color: tuple = (255, 255, 0),
    stroke_color: tuple = (0, 0, 0),
    stroke_width: int = 3,
    max_width: Optional[int] = None,
    duration: float = 1.0,
    start_time: float = 0.0,
    position: tuple = ('center', 'center')
):
    """
    Create a MoviePy ImageClip from Pillow-rendered text.
    Replacement for TextClip that's 10-50x faster.
    """
    from moviepy import ImageClip
    
    # Render text to numpy array
    text_array = _render_text_to_array(
        text=text,
        font_path=font_path,
        font_size=font_size,
        text_color=text_color,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        max_width=max_width
    )
    
    # Create ImageClip from the array
    clip = ImageClip(text_array, is_mask=False, transparent=True)
    clip = clip.with_duration(duration).with_start(start_time).with_position(position)
    
    return clip


# --- Animated Caption System ---
# These functions create TikTok-style animated captions with:
# - Pop-in animations (scale from small to full size)
# - Keyword highlighting (different colors for emphasis words)
# - Multi-word grouping (2-4 words at a time)
# - Emoji integration

def _ease_out_back(t: float) -> float:
    """Easing function for bounce/overshoot effect. t in [0,1] -> output in [0, ~1.1]"""
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * pow(t - 1, 3) + c1 * pow(t - 1, 2)


def _ease_out_elastic(t: float) -> float:
    """Elastic bounce easing for more dramatic pop effect."""
    import math
    if t == 0:
        return 0
    if t == 1:
        return 1
    p = 0.3
    return pow(2, -10 * t) * math.sin((t - p / 4) * (2 * math.pi) / p) + 1


def _render_animated_caption_frame(
    t: float,
    words: list,  # List of dicts: {word, is_emphasis, emoji}
    font_path: Optional[str],
    base_font_size: int,
    animation_duration: float = 0.15,
    target_width: int = TARGET_WIDTH,
    target_height: int = TARGET_HEIGHT
):
    """
    Render a single frame of animated caption.
    Returns numpy array (H, W, 4) with RGBA.
    
    Animation: Scale from 0.7 -> 1.0 with bounce easing over animation_duration.
    """
    from PIL import Image, ImageDraw
    import numpy as np
    
    # Calculate animation progress
    if t < animation_duration:
        progress = t / animation_duration
        scale = 0.7 + 0.3 * _ease_out_back(progress)
        scale = min(scale, 1.1)  # Cap overshoot
    else:
        scale = 1.0
    
    # Create canvas
    canvas_w = target_width
    canvas_h = int(target_height * 0.15)  # Caption area height
    img = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Build the caption text with styling
    # Normal words: Yellow (#FFFF00)
    # Emphasis words: Cyan (#00FFFF) or Green (#00FF00)
    # Emojis: Rendered inline
    
    # Calculate total text width first for centering
    scaled_font_size = int(base_font_size * scale)
    emphasis_font_size = int(scaled_font_size * 1.15)  # Emphasis words 15% larger
    
    font_normal = _get_pillow_font(font_path, scaled_font_size)
    font_emphasis = _get_pillow_font(font_path, emphasis_font_size)
    
    # Calculate total width
    total_width = 0
    word_widths = []
    space_width = draw.textbbox((0, 0), " ", font=font_normal)[2]
    
    for i, word_info in enumerate(words):
        word = word_info.get("word", "").upper()
        is_emphasis = word_info.get("is_emphasis", False)
        emoji = word_info.get("emoji")
        
        font = font_emphasis if is_emphasis else font_normal
        bbox = draw.textbbox((0, 0), word, font=font, stroke_width=3)
        w = bbox[2] - bbox[0]
        word_widths.append((w, word, is_emphasis, emoji, font))
        total_width += w
        if i < len(words) - 1:
            total_width += space_width
    
    # Center position
    x_start = (canvas_w - total_width) // 2
    y_center = canvas_h // 2
    
    # Draw each word
    x = x_start
    for w, word, is_emphasis, emoji, font in word_widths:
        # Color based on emphasis
        if is_emphasis:
            text_color = (0, 255, 255, 255)  # Cyan for emphasis
        else:
            text_color = (255, 255, 0, 255)  # Yellow for normal
        
        # Get vertical position (center text vertically)
        bbox = draw.textbbox((0, 0), word, font=font, stroke_width=3)
        text_height = bbox[3] - bbox[1]
        y = y_center - text_height // 2
        
        # Draw with stroke (outline)
        draw.text(
            (x, y),
            word,
            font=font,
            fill=text_color,
            stroke_width=3,
            stroke_fill=(0, 0, 0, 255)
        )
        
        # Draw emoji if present (to the right of the word)
        if emoji:
            try:
                emoji_bbox = draw.textbbox((0, 0), emoji, font=font_normal)
                emoji_w = emoji_bbox[2] - emoji_bbox[0]
                draw.text((x + w + 5, y), emoji, font=font_normal, fill=(255, 255, 255, 255))
                x += emoji_w + 5
            except:
                pass  # Skip emoji if rendering fails
        
        x += w + space_width
    
    return np.array(img)


def _create_animated_caption_clip(
    words: list,  # List of dicts: {word, is_emphasis, emoji}
    font_path: Optional[str],
    base_font_size: int,
    duration: float,
    start_time: float,
    target_width: int = TARGET_WIDTH,
    target_height: int = TARGET_HEIGHT,
    y_position: float = 0.55  # Vertical position (0-1, from top)
):
    """
    Create a caption clip for multi-word groups.
    OPTIMIZED: Uses static ImageClip instead of per-frame VideoClip rendering.
    This is 50-100x faster than the animated per-frame approach.
    """
    from moviepy import ImageClip
    
    # Render the static frame (final state of animation)
    static_frame = _render_animated_caption_frame(
        t=1.0,  # Fully rendered state
        words=words,
        font_path=font_path,
        base_font_size=base_font_size,
        target_width=target_width,
        target_height=target_height
    )
    
    # Create static ImageClip (MUCH faster than VideoClip with make_frame)
    clip = ImageClip(static_frame, transparent=True)
    clip = clip.with_duration(duration)
    clip = clip.with_start(start_time)
    
    # Position the caption
    y_pos = int(target_height * y_position)
    clip = clip.with_position(('center', y_pos))
    
    return clip


def _group_words_for_captions(
    relevant_words: list,
    words_per_group: int = 3
) -> list:
    """
    Group words into caption groups of 2-4 words each.
    Returns list of {words: [...], start_time, end_time}
    """
    groups = []
    current_group = []
    group_start = None
    group_end = None
    
    for word_info in relevant_words:
        if len(current_group) == 0:
            group_start = word_info["start"]
        
        current_group.append({
            "word": word_info["word"],
            "is_emphasis": False,  # Will be enhanced by LLM later
            "emoji": None
        })
        group_end = word_info["end"]
        
        if len(current_group) >= words_per_group:
            groups.append({
                "words": current_group,
                "start_time": group_start,
                "end_time": group_end
            })
            current_group = []
            group_start = None
    
    # Don't forget the last group
    if current_group:
        groups.append({
            "words": current_group,
            "start_time": group_start,
            "end_time": group_end
        })
    
    return groups


# --- Zoom & Ken Burns Effects ---
# Dynamic zoom effects based on energy level for professional look
# OPTIMIZED: Uses static crop+resize instead of per-frame Python transforms

def _apply_zoom_effect(
    clip,
    energy_level: str = "medium",
    is_hook: bool = False
):
    """
    Apply zoom effect to a clip based on energy level.
    OPTIMIZED: Uses static center-crop + resize instead of per-frame transforms.
    This is 50-100x faster than the per-frame PIL approach.
    
    Effects (applied as static zoom):
    - hook: 1.08x zoom
    - climax: 1.12x zoom
    - high: 1.06x zoom
    - medium: 1.03x zoom
    - low: No zoom
    """
    # Define zoom levels based on energy
    zoom_configs = {
        "climax": 1.12,
        "high": 1.06,
        "medium": 1.03,
        "low": 1.0,
    }
    
    # Hook gets special treatment
    if is_hook:
        zoom = 1.08
    else:
        zoom = zoom_configs.get(energy_level, 1.0)
    
    # No zoom needed
    if zoom <= 1.0:
        return clip
    
    # Apply static zoom via crop + resize (FAST)
    w, h = clip.w, clip.h
    new_w = int(w / zoom)
    new_h = int(h / zoom)
    
    # Center crop
    x1 = (w - new_w) // 2
    y1 = (h - new_h) // 2
    
    # Crop and resize back to original dimensions
    return clip.cropped(x1=x1, y1=y1, x2=x1 + new_w, y2=y1 + new_h).resized((w, h))


def _apply_speed_ramp(
    clip,
    energy_level: str = "medium"
):
    """
    Apply subtle speed ramping based on energy level.
    
    - climax: Slight slow-mo (0.92x) for dramatic effect
    - high: Normal speed (1.0x)
    - medium: Normal speed (1.0x)
    - low: Slight speedup (1.05x) to keep pace
    """
    speed_configs = {
        "climax": 0.92,   # Slight slow-mo for impact
        "high": 1.0,
        "medium": 1.0,
        "low": 1.05,      # Slightly faster for pacing
    }
    
    speed = speed_configs.get(energy_level, 1.0)
    
    if speed == 1.0:
        return clip
    
    return clip.with_speed_scaled(speed)


# --- Background Music & Audio Ducking ---
# Add background music that automatically ducks when speech is detected

# Directory for background music tracks
MUSIC_DIR = "assets/music"

def _get_available_music_tracks() -> List[str]:
    """Get list of available background music files."""
    if not os.path.exists(MUSIC_DIR):
        os.makedirs(MUSIC_DIR, exist_ok=True)
        return []
    
    music_extensions = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
    tracks = []
    for f in os.listdir(MUSIC_DIR):
        if os.path.splitext(f)[1].lower() in music_extensions:
            tracks.append(os.path.join(MUSIC_DIR, f))
    return tracks


def _select_music_for_energy(energy_levels: List[str]) -> Optional[str]:
    """
    Select appropriate background music based on video energy levels.
    Returns path to music file or None if no music available.
    
    For now, just returns the first available track.
    Future: match music BPM/mood to video energy.
    """
    tracks = _get_available_music_tracks()
    if not tracks:
        return None
    
    # Simple selection - use first track
    # TODO: Implement mood matching based on energy_levels
    return tracks[0]


def _create_ducking_envelope(
    speech_segments: List[dict],
    total_duration: float,
    sample_rate: int = 44100,
    duck_level: float = 0.25,      # Volume during speech (0.25 = 25%)
    attack_time: float = 0.1,       # Fade down time
    release_time: float = 0.3      # Fade up time
):
    """
    Create a volume envelope array for ducking music under speech.
    
    Returns numpy array of volume multipliers (0-1) for each sample.
    """
    import numpy as np
    
    total_samples = int(total_duration * sample_rate)
    envelope = np.ones(total_samples)  # Start at full volume
    
    attack_samples = int(attack_time * sample_rate)
    release_samples = int(release_time * sample_rate)
    
    for seg in speech_segments:
        start_sample = int(seg["start"] * sample_rate)
        end_sample = int(seg["end"] * sample_rate)
        
        # Clamp to valid range
        start_sample = max(0, min(start_sample, total_samples))
        end_sample = max(0, min(end_sample, total_samples))
        
        if start_sample >= end_sample:
            continue
        
        # Attack (fade down before speech)
        attack_start = max(0, start_sample - attack_samples)
        for i in range(attack_start, start_sample):
            progress = (i - attack_start) / attack_samples if attack_samples > 0 else 1
            target_vol = 1.0 - progress * (1.0 - duck_level)
            envelope[i] = min(envelope[i], target_vol)
        
        # Duck during speech
        envelope[start_sample:end_sample] = duck_level
        
        # Release (fade up after speech)
        release_end = min(total_samples, end_sample + release_samples)
        for i in range(end_sample, release_end):
            progress = (i - end_sample) / release_samples if release_samples > 0 else 1
            target_vol = duck_level + progress * (1.0 - duck_level)
            envelope[i] = min(envelope[i], target_vol)
    
    return envelope


def _apply_ducking_to_audio(
    music_clip,
    envelope,
    sample_rate: int = 44100
):
    """
    Apply ducking envelope to music audio clip.
    Returns modified audio clip.
    """
    import numpy as np
    from moviepy import AudioClip
    
    original_audio = music_clip.audio
    if original_audio is None:
        return music_clip
    
    # Get audio as array
    fps = original_audio.fps or sample_rate
    
    def make_frame(t):
        # Get original audio frame
        original_frame = original_audio.get_frame(t)
        
        # Calculate envelope index
        if isinstance(t, np.ndarray):
            indices = (t * sample_rate).astype(int)
            indices = np.clip(indices, 0, len(envelope) - 1)
            vol = envelope[indices]
            # Handle stereo/mono
            if len(original_frame.shape) > 1:
                vol = vol.reshape(-1, 1)
        else:
            idx = int(t * sample_rate)
            idx = max(0, min(idx, len(envelope) - 1))
            vol = envelope[idx]
        
        return original_frame * vol
    
    ducked_audio = AudioClip(make_frame, duration=music_clip.duration, fps=fps)
    return music_clip.with_audio(ducked_audio)


def _mix_audio_tracks(
    original_audio,
    music_audio,
    music_volume: float = 0.3  # Base music volume (before ducking)
):
    """
    Mix original video audio with background music.
    Returns combined AudioClip.
    """
    from moviepy import CompositeAudioClip
    
    if original_audio is None:
        return music_audio.with_volume_scaled(music_volume) if music_audio else None
    
    if music_audio is None:
        return original_audio
    
    # Scale music volume
    music_scaled = music_audio.with_volume_scaled(music_volume)
    
    # Composite the audio tracks
    return CompositeAudioClip([original_audio, music_scaled])


def _add_background_music(
    video_clip,
    speech_segments: List[dict],
    music_path: Optional[str] = None,
    music_volume: float = 0.3,
    enable_ducking: bool = True
):
    """
    Add background music to video with automatic ducking during speech.
    
    Args:
        video_clip: The video clip to add music to
        speech_segments: List of {start, end} dicts for speech timing
        music_path: Path to music file (or None to auto-select)
        music_volume: Base volume for music (0-1)
        enable_ducking: Whether to duck music during speech
    
    Returns:
        Video clip with background music added
    """
    from moviepy import AudioFileClip, CompositeAudioClip
    
    # Select music track
    if music_path is None:
        music_path = _select_music_for_energy([])
    
    if music_path is None or not os.path.exists(music_path):
        print("[Music] No background music available, skipping...")
        return video_clip
    
    try:
        print(f"[Music] Adding background music: {os.path.basename(music_path)}")
        
        # Load music and loop/trim to video duration
        music_clip = AudioFileClip(music_path)
        video_duration = video_clip.duration
        
        # Loop music if shorter than video
        if music_clip.duration < video_duration:
            loops_needed = int(video_duration / music_clip.duration) + 1
            from moviepy import concatenate_audioclips
            music_clip = concatenate_audioclips([music_clip] * loops_needed)
        
        # Trim to video duration
        music_clip = music_clip.subclipped(0, video_duration)
        
        # Apply ducking if enabled
        if enable_ducking and speech_segments:
            print(f"[Music] Applying audio ducking ({len(speech_segments)} speech segments)")
            envelope = _create_ducking_envelope(
                speech_segments=speech_segments,
                total_duration=video_duration,
                duck_level=0.2,  # Duck to 20% during speech
                attack_time=0.15,
                release_time=0.4
            )
            # Apply envelope to music
            music_audio = music_clip
            # Scale music volume
            music_audio = music_audio.with_volume_scaled(music_volume)
            
            # Create ducked version using numpy
            import numpy as np
            
            def apply_envelope(get_frame, t):
                frame = get_frame(t)
                if isinstance(t, np.ndarray):
                    indices = (t * 44100).astype(int)
                    indices = np.clip(indices, 0, len(envelope) - 1)
                    vol = envelope[indices]
                    if len(frame.shape) > 1:
                        vol = vol.reshape(-1, 1)
                else:
                    idx = int(t * 44100)
                    idx = max(0, min(idx, len(envelope) - 1))
                    vol = envelope[idx]
                return frame * vol
            
            from moviepy import AudioClip
            ducked_music = AudioClip(
                lambda t: apply_envelope(music_audio.get_frame, t),
                duration=video_duration,
                fps=music_audio.fps or 44100
            )
        else:
            ducked_music = music_clip.with_volume_scaled(music_volume)
        
        # Mix with original audio
        original_audio = video_clip.audio
        if original_audio is not None:
            mixed_audio = CompositeAudioClip([original_audio, ducked_music])
            video_clip = video_clip.with_audio(mixed_audio)
            print("[Music] Background music mixed with ducking")
        else:
            video_clip = video_clip.with_audio(ducked_music)
            print("[Music] Background music added (no original audio)")
        
        return video_clip
        
    except Exception as e:
        print(f"[Music] Error adding background music: {e}")
        return video_clip

# --- Logging Setup ---
class StructuredLogger:
    def __init__(self, log_file="execution_logs.jsonl"):
        self.log_file = log_file
        self.run_id = str(uuid.uuid4())
        self.log_event("run_start", {"timestamp": datetime.datetime.now().isoformat()})

    def log_event(self, event_type: str, data: Dict[str, Any]):
        entry = {
            "run_id": self.run_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "event_type": event_type,
            "data": data
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

# Global logger instance
logger = StructuredLogger()

# --- State Definition ---
class VideoState(TypedDict):
    input_video_path: str
    audio_path: str
    transcript_text: str
    transcript_segments: List[dict] # List of {start: float, end: float, text: str}
    cuts: List[dict] # {start: float, end: float, reason: str}
    heading: str
    output_video_path: str
    dev_mode: bool  # Development mode: cache transcriptions
    export_mode: str  # "preview" or "production" - controls output quality
    use_critic: bool  # Whether to run critic agent for quality review
    use_music: bool  # Whether to add background music
    parameters: Dict[str, Any] # For extensibility/future use

# --- Helper Functions ---

def get_video_hash(video_path: str) -> str:
    """Calculate MD5 hash of video file for cache identification."""
    hash_md5 = hashlib.md5()
    with open(video_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()[:16]

# --- Nodes ---

def extract_audio(state: VideoState):
    """Extracts audio from the input video."""
    print("--- Extracting Audio ---")
    logger.log_event("node_start", {"node": "extract_audio", "input": {"video_path": state["input_video_path"]}})
    
    video_path = state["input_video_path"]
    audio_path = "temp_audio.mp3"
    
    try:
        video = VideoFileClip(video_path)
        if video.audio is None:
            raise ValueError("Video has no audio track")
        video.audio.write_audiofile(audio_path, logger=None)
        video.close()
        
        result = {"audio_path": audio_path}
        logger.log_event("node_end", {"node": "extract_audio", "output": result})
        return result
    except Exception as e:
        print(f"Error extracting audio: {e}")
        logger.log_event("node_error", {"node": "extract_audio", "error": str(e)})
        return {"audio_path": None} 

def transcribe_audio(state: VideoState):
    """Transcribes audio using local Whisper model. Always caches transcripts by filename."""
    print("--- Transcribing Audio ---")
    logger.log_event("node_start", {"node": "transcribe_audio", "input": {"audio_path": state["audio_path"]}})
    
    audio_path = state["audio_path"]
    video_path = state["input_video_path"]
    
    if not audio_path or not os.path.exists(audio_path):
        error_msg = "Audio file not found or extraction failed."
        logger.log_event("node_error", {"node": "transcribe_audio", "error": error_msg})
        raise FileNotFoundError(error_msg)

    # Always-on cache: keyed by video filename (assumed unique)
    os.makedirs(CACHE_DIR, exist_ok=True)
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    cache_file = os.path.join(CACHE_DIR, f"{video_basename}_transcript.json")
    
    if os.path.exists(cache_file):
        print(f"[CACHE] Loading cached transcript: {cache_file}")
        logger.log_event("cache_hit", {"node": "transcribe_audio", "cache_file": cache_file})
        with open(cache_file, "r", encoding="utf-8") as f:
            cached = json.load(f)
        logger.log_event("node_end", {"node": "transcribe_audio", "output": {"transcript_text_preview": cached["transcript_text"][:100], "segment_count": len(cached["transcript_segments"]), "from_cache": True}})
        return cached

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        model = whisper.load_model("base", device=device)
        result = model.transcribe(audio_path, word_timestamps=True)
        
        output = {
            "transcript_text": result["text"],
            "transcript_segments": result["segments"]
        }
        
        # Always save to cache
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False)
        print(f"[CACHE] Saved transcript: {cache_file}")
        logger.log_event("cache_save", {"node": "transcribe_audio", "cache_file": cache_file})
        
        logger.log_event("node_end", {"node": "transcribe_audio", "output": {"transcript_text_preview": result["text"][:100], "segment_count": len(result["segments"])}})
        return output
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        logger.log_event("node_error", {"node": "transcribe_audio", "error": str(e)})
        raise e

def analyze_transcript(state: VideoState):
    """Analyzes transcript using OpenRouter/LLM to find viral cuts with hook optimization and energy levels."""
    print("--- Analyzing Transcript (Enhanced) ---")
    logger.log_event("node_start", {"node": "analyze_transcript", "input": {"transcript_preview": state["transcript_text"][:200]}})
    
    transcript_text = state["transcript_text"]
    segments = state["transcript_segments"]
    
    # Initialize LLM via ModelFactory
    llm = ModelFactory.get_model(
        provider=os.getenv("LLM_PROVIDER", "openrouter"),
        model_name=os.getenv("LLM_MODEL", "z-ai/glm-4.5-air:free"),
        temperature=0.7
    )
    
    # Detailed Context for the LLM
    segment_details = "\n".join([f"[{s['start']:.2f}-{s['end']:.2f}]: {s['text']}" for s in segments])
    
    system_prompt = """
You are an ELITE video editor creating VIRAL shorts for Gen Z on TikTok/Reels/Shorts.
Your goal is to create content that HOOKS viewers in the first 3 seconds and keeps them watching.

## HOOK OPTIMIZATION (CRITICAL)
The first segment MUST be the most attention-grabbing moment. Look for:
- Shocking statements, surprising facts, or bold claims
- Questions that create curiosity gaps
- High-energy moments or emotional peaks
- Controversial or unexpected content
- "Wait what?" moments

Mark the hook segment with "is_hook": true.

## ENERGY LEVELS
For each segment, specify the energy level for dynamic editing effects:
- "climax": Peak moment, most intense (use for 1-2 segments max)
- "high": High energy, exciting content
- "medium": Normal pacing
- "low": Calm, setup, or transition moments

## TRANSITIONS
- "cut": Fast-paced (use for 80% of transitions)
- "crossfade": Smooth flow between topics  
- "fade_to_black": Dramatic pause before reveal

## RULES
1. Total video length: 30-60 seconds
2. Hook MUST grab attention in first 3 seconds
3. Keep 3-7 cuts total
4. The hook segment should START the video (order[0] should be hook_segment_index)
5. Build tension -> climax -> resolution arc when possible

CRITICAL: Respond with ONLY valid JSON matching this EXACT schema:
{
  "cuts": [
    {"start": 45.0, "end": 48.5, "reason": "Shocking reveal - perfect hook", "transition": "cut", "is_hook": true, "energy_level": "high"},
    {"start": 10.5, "end": 18.2, "reason": "Context setup", "transition": "crossfade", "is_hook": false, "energy_level": "medium"},
    {"start": 50.0, "end": 58.0, "reason": "Climax moment", "transition": "cut", "is_hook": false, "energy_level": "climax"}
  ],
  "order": [0, 1, 2],
  "hook_segment_index": 0
}
"""
    
    user_message = f"Here is the video transcript. Find the HOOK first, then build the narrative:\n{segment_details}"
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
    
    logger.log_event("llm_call", {
        "node": "analyze_transcript", 
        "system_prompt": system_prompt, 
        "user_message_preview": user_message[:500] + "..."
    })

    try:
        # Use robust structured output
        result: AnalysisResult = call_llm_with_structure(llm, messages, AnalysisResult)
        
        logger.log_event("llm_response", {"node": "analyze_transcript", "structured_output": result.dict()})
        
        # Convert Pydantic model to dict for state
        cuts_data = [cut.dict() for cut in result.cuts]
        
        # Reorder if 'order' is provided (hook should be first!)
        if result.order:
            ordered_cuts = [cuts_data[i] for i in result.order if i < len(cuts_data)]
        else:
            # If no order provided, ensure hook is first
            hook_cuts = [c for c in cuts_data if c.get("is_hook")]
            non_hook_cuts = [c for c in cuts_data if not c.get("is_hook")]
            ordered_cuts = hook_cuts + non_hook_cuts
        
        # Log hook info
        hook_cut = next((c for c in ordered_cuts if c.get("is_hook")), None)
        if hook_cut:
            print(f"[HOOK] Selected hook: {hook_cut['start']:.1f}s - {hook_cut['end']:.1f}s - {hook_cut['reason']}")
        
        logger.log_event("node_end", {"node": "analyze_transcript", "output": {"cuts": ordered_cuts, "hook": hook_cut}})
        return {"cuts": ordered_cuts}
        
    except Exception as e:
        print(f"Error analyzing transcript (retries failed): {e}")
        logger.log_event("node_error", {"node": "analyze_transcript", "error": str(e)})
        # Fallback: Just take the first 30 seconds if Analysis fails
        return {"cuts": [{"start": 0, "end": 30, "reason": "Fallback - LLM Failed", "transition": "cut", "is_hook": True, "energy_level": "medium"}]}

def generate_heading(state: VideoState):
    """Generates a viral, witty heading for the video using LLM."""
    print("--- Generating Heading ---")
    logger.log_event("node_start", {"node": "generate_heading", "input": {"transcript_preview": state["transcript_text"][:200]}})
    
    transcript_text = state["transcript_text"]
    
    llm = ModelFactory.get_model(
        provider=os.getenv("LLM_PROVIDER", "openrouter"),
        model_name=os.getenv("LLM_MODEL", "z-ai/glm-4.5-air:free"),
        temperature=0.9
    )
    
    system_prompt = """
You are a video editor creating context overlays for short-form content on TikTok and Instagram Reels.
Your job is to write a SINGLE, concise heading that gives the viewer BACKGROUND CONTEXT about what is happening in the video.

Rules:
- Provide factual context: WHO is speaking, WHERE, and WHAT the situation is about
- MAX 8-12 words
- NO hashtags, NO emojis
- Should read like a news caption or scene description
- Do NOT use meme formats like "POV:", "Me when:", "Nobody:" etc.
- Be specific, not generic

CRITICAL: Respond with ONLY valid JSON in this EXACT format:
{"heading": "Your contextual heading here"}

Examples:
{"heading": "Student testifies before Congress on rising tuition costs"}
{"heading": "CEO explains why layoffs were necessary at town hall"}
{"heading": "Doctor breaks down the real risk behind viral health trend"}
"""
    
    user_message = f"Here is the video transcript:\n{transcript_text[:2000]}..." # Truncate for efficiency if needed
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
    
    logger.log_event("llm_call", {
        "node": "generate_heading",
        "system_prompt": system_prompt, 
        "user_message_preview": user_message[:500]
    })
    
    try:
        # Use robust structured output
        result: HeadingResult = call_llm_with_structure(llm, messages, HeadingResult)
        
        heading = result.heading
        print(f"Generated Heading: {heading}")
        
        logger.log_event("node_end", {"node": "generate_heading", "output": {"heading": heading}})
        return {"heading": heading}
        
    except Exception as e:
        print(f"Error generating heading (retries failed): {e}")
        logger.log_event("node_error", {"node": "generate_heading", "error": str(e)})
        return {"heading": "Economics 101"} # Fallback

def _resolve_font():
    """Cross-platform font resolution with fallback to None (ImageMagick default)."""
    font_candidates = [
        # Windows
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        # Linux — Debian/Ubuntu
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        # Linux — Fedora/RHEL
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in font_candidates:
        if os.path.exists(candidate):
            return candidate
    print("Warning: No system font found, using default ImageMagick font.")
    return None


def _detect_face_offset(clip, new_w, src_w):
    """Detect face position and return x_offset for cropping. Returns None if no face found."""
    try:
        import cv2
        face_cascade = _get_face_cascade()
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


def _crop_to_vertical(clip, start_time=None, face_x_offset=None, target_w=TARGET_WIDTH, target_h=TARGET_HEIGHT):
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


def _get_relevant_words_for_cuts(transcript_segments: List[dict], cuts: List[dict]) -> List[dict]:
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


# --- Vision LLM Integration ---
# Analyze video keyframes using vision-capable LLMs for better context understanding

def _extract_keyframes(
    video_path: str,
    num_frames: int = 5,
    uniform: bool = True
) -> List[str]:
    """
    Extract keyframes from video and save as temporary images.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        uniform: If True, extract uniformly distributed frames
    
    Returns:
        List of paths to extracted frame images
    """
    import numpy as np
    from PIL import Image
    
    frame_paths: List[str] = []
    cache_dir = os.path.join(CACHE_DIR, "keyframes")
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        
        # Calculate frame times
        if uniform:
            times = np.linspace(0.1, duration - 0.1, num_frames)
        else:
            # Focus on beginning (hook), middle, and end
            times = [
                duration * 0.05,   # Near start (hook)
                duration * 0.25,   # First quarter
                duration * 0.5,    # Middle
                duration * 0.75,   # Third quarter
                duration * 0.95,   # Near end
            ][:num_frames]
        
        for i, t in enumerate(times):
            frame = clip.get_frame(t)
            img = Image.fromarray(frame)
            
            # Resize for efficient API calls (max 1024px)
            max_dim = 1024
            if max(img.size) > max_dim:
                ratio = max_dim / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            frame_path = os.path.join(cache_dir, f"keyframe_{i}.jpg")
            img.save(frame_path, "JPEG", quality=85)
            frame_paths.append(frame_path)
        
        clip.close()
        print(f"[Vision] Extracted {len(frame_paths)} keyframes")
        return frame_paths
        
    except Exception as e:
        print(f"[Vision] Error extracting keyframes: {e}")
        return []


def _encode_image_base64(image_path: str) -> str:
    """Encode image file to base64 string."""
    import base64
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _analyze_keyframes_with_vision(
    frame_paths: List[str],
    transcript_preview: str = ""
) -> Dict[str, Any]:
    """
    Analyze keyframes using a vision LLM to understand visual context.
    
    Returns dict with:
        - scene_description: Overall description of the video
        - visual_elements: Key visual elements detected
        - suggested_hooks: Visual moments that could be hooks
        - mood: Detected mood/tone
        - b_roll_detected: Whether B-roll footage is present
    """
    if not frame_paths:
        return {"error": "No frames to analyze"}
    
    try:
        # Get vision model
        vision_provider = os.getenv("VISION_PROVIDER", "openrouter")
        vision_model = os.getenv("VISION_MODEL", "openai/gpt-4o-mini")
        
        llm = ModelFactory.get_model(
            provider=vision_provider,
            model_name=vision_model,
            temperature=0.3
        )
        
        # Build multimodal message with images
        image_contents = []
        for path in frame_paths[:4]:  # Limit to 4 frames to control costs
            b64 = _encode_image_base64(path)
            image_contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
        
        system_prompt = """You are a video analysis expert helping create viral short-form content.
Analyze these keyframes from a video and provide insights that will help edit it into an engaging short.

Respond with JSON in this exact format:
{
    "scene_description": "Brief description of what's happening in the video",
    "visual_elements": ["list", "of", "key", "visual", "elements"],
    "visual_hooks": ["moments that would grab attention visually"],
    "mood": "detected mood/tone (e.g., serious, funny, dramatic, educational)",
    "has_broll": true/false,
    "speaker_visible": true/false,
    "suggested_emoji": ["relevant", "emojis"],
    "color_mood": "warm/cool/neutral/vibrant"
}"""

        text_content = f"""Analyze these video keyframes.

Transcript preview: {transcript_preview[:500] if transcript_preview else 'Not available'}

What visual elements, mood, and hook-worthy moments do you see?"""

        # Create message with images and text
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {"type": "text", "text": text_content},
                *image_contents
            ])
        ]
        
        logger.log_event("llm_call", {
            "node": "vision_analysis",
            "model": vision_model,
            "num_frames": len(frame_paths)
        })
        
        response = llm.invoke(messages)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Parse JSON response
        import re
        
        # Strip markdown if present
        content_str = str(content).strip()
        if content_str.startswith("```"):
            match = re.search(r'```(?:json)?\s*(.*?)\s*```', content_str, re.DOTALL)
            if match:
                content_str = match.group(1)
        
        result = json.loads(content_str)
        
        logger.log_event("llm_response", {
            "node": "vision_analysis",
            "result": result
        })
        
        print(f"[Vision] Analysis complete: {result.get('mood', 'unknown')} mood, "
              f"{len(result.get('visual_elements', []))} visual elements detected")
        
        return result
        
    except Exception as e:
        print(f"[Vision] Error analyzing keyframes: {e}")
        logger.log_event("node_error", {"node": "vision_analysis", "error": str(e)})
        return {"error": str(e)}


def analyze_video_visually(state: VideoState) -> dict:
    """
    LangGraph node to analyze video visually using Vision LLM.
    Extracts keyframes and analyzes them for visual context.
    """
    print("--- Analyzing Video Visually ---")
    logger.log_event("node_start", {"node": "analyze_visually"})
    
    video_path = state["input_video_path"]
    transcript = state.get("transcript_text", "")
    
    # Check if vision analysis is enabled
    enable_vision = os.getenv("ENABLE_VISION_ANALYSIS", "true").lower() == "true"
    
    if not enable_vision:
        print("[Vision] Vision analysis disabled via ENABLE_VISION_ANALYSIS env var")
        return {"parameters": {**state.get("parameters", {}), "vision_analysis": None}}
    
    # Extract keyframes
    frame_paths = _extract_keyframes(video_path, num_frames=5)
    
    if not frame_paths:
        print("[Vision] No keyframes extracted, skipping vision analysis")
        return {"parameters": {**state.get("parameters", {}), "vision_analysis": None}}
    
    # Analyze with vision LLM
    analysis = _analyze_keyframes_with_vision(frame_paths, transcript)
    
    # Store in parameters for later use
    updated_params = {**state.get("parameters", {}), "vision_analysis": analysis}
    
    logger.log_event("node_end", {"node": "analyze_visually", "output": analysis})
    
    return {"parameters": updated_params}


def edit_video(state: VideoState):
    """Cuts and stitches the video with 9:16 vertical format, subtitles, transitions, and context heading.
    
    PERFORMANCE OPTIMIZATIONS:
    - Pre-filters transcript segments to only relevant words
    - Caches face detection (runs once per video, not per cut)
    - Uses Pillow for text rendering (10-50x faster than TextClip/ImageMagick)
    - Uses optimized FFmpeg encoding settings (threads, preset)
    - Flattened composition structure
    """
    print("--- Editing Video ---")
    logger.log_event("node_start", {"node": "edit_video", "input": {"cuts_count": len(state["cuts"]), "heading": state.get("heading")}})
    
    video_path = state["input_video_path"]
    cuts = state["cuts"]
    output_path = "output.mp4"
    
    # --- Export mode settings ---
    export_mode = state.get("export_mode", "preview")
    if export_mode == "production":
        target_width = 1080
        target_height = 1920
        print("🎬 [PRODUCTION] Exporting at 1080x1920")
    else:
        target_width = TARGET_WIDTH  # 512
        target_height = TARGET_HEIGHT  # 720
        print("⚡ [PREVIEW] Exporting at 512x720")
    
    target_aspect = target_width / target_height
    
    # --- 30-second duration limit for short-form content ---
    MAX_DURATION = 30.0
    total_duration = sum(max(0, cut["end"] - cut["start"]) for cut in cuts)
    
    if total_duration > MAX_DURATION:
        print(f"⚠️ Total duration ({total_duration:.1f}s) exceeds {MAX_DURATION}s limit, trimming cuts...")
        trimmed_cuts = []
        accumulated = 0.0
        for cut in cuts:
            cut_duration = cut["end"] - cut["start"]
            if accumulated + cut_duration <= MAX_DURATION:
                trimmed_cuts.append(cut)
                accumulated += cut_duration
            elif accumulated < MAX_DURATION:
                # Partially include this cut
                remaining = MAX_DURATION - accumulated
                trimmed_cuts.append({**cut, "end": cut["start"] + remaining})
                accumulated = MAX_DURATION
                break
        cuts = trimmed_cuts
        print(f"✅ Trimmed to {len(cuts)} cuts ({accumulated:.1f}s total)")
    
    # MoviePy imports
    from moviepy.video import fx as vfx
    from moviepy import CompositeVideoClip, VideoFileClip
    import numpy as np

    try:
        original_clip = VideoFileClip(video_path)
        
        font_path = _resolve_font()

        # --- Font sizes proportional to target output width ---
        subtitle_font_size = max(16, int(target_width * 0.035))   # ~38px on 1080w
        heading_font_size = max(18, int(target_width * 0.04))     # ~43px on 1080w
        subtitle_stroke = max(1, int(target_width * 0.002))       # ~2px
        heading_stroke = max(1, int(target_width * 0.0025))       # ~3px

        # --- OPTIMIZATION 1: Pre-filter transcript to only relevant words ---
        print("[Performance] Pre-filtering transcript segments...")
        relevant_words = _get_relevant_words_for_cuts(state["transcript_segments"], cuts)
        print(f"[Performance] Filtered to {len(relevant_words)} relevant words (from full transcript)")

        # --- OPTIMIZATION 2: Detect face ONCE on the original video ---
        src_w, src_h = original_clip.w, original_clip.h
        src_aspect = src_w / src_h
        face_x_offset = None
        if src_aspect > target_aspect:
            new_w = int(src_h * target_aspect)
            # Run face detection once on a sample from the middle of the video
            sample_clip = original_clip.subclipped(original_clip.duration * 0.4, min(original_clip.duration * 0.6, original_clip.duration))
            face_x_offset = _detect_face_offset(sample_clip, new_w, src_w)
            if face_x_offset is None:
                face_x_offset = (src_w - new_w) // 2  # Fall back to center
                print("[Performance] Using center crop (no face detected)")
            else:
                print(f"[Performance] Face detected once, reusing offset={face_x_offset} for all cuts")

        # --- OPTIMIZATION 3: Create ANIMATED CAPTIONS with multi-word groups ---
        # Using Pillow-based rendering with pop-in animations
        all_subtitle_clips = []
        video_clips = []
        cumulative_time = 0.0  # Track position in final video timeline
        
        # Group words for display (2-4 words per group for readability)
        caption_groups = _group_words_for_captions(relevant_words, words_per_group=3)
        print(f"[Captions] Created {len(caption_groups)} caption groups from {len(relevant_words)} words")
        
        print("[Performance] Rendering ANIMATED captions with Pillow...")
        for cut in cuts:
            start = max(0, cut["start"])
            end = min(original_clip.duration, cut["end"])
            energy_level = cut.get("energy_level", "medium")
            is_hook = cut.get("is_hook", False)
            
            if end > start:
                clip = original_clip.subclipped(start, end)
                clip = _crop_to_vertical(clip, start, face_x_offset, target_width, target_height)
                
                # --- Apply Zoom & Ken Burns Effects based on energy ---
                clip = _apply_zoom_effect(clip, energy_level=energy_level, is_hook=is_hook)
                
                # --- Apply Speed Ramping for dramatic moments ---
                clip = _apply_speed_ramp(clip, energy_level=energy_level)
                
                clip_duration = clip.duration
                
                # Find caption groups that overlap with this cut
                for group in caption_groups:
                    group_start = group["start_time"]
                    group_end = group["end_time"]
                    
                    overlap_start = max(start, group_start)
                    overlap_end = min(end, group_end)
                    
                    if overlap_end > overlap_start:
                        # Calculate position relative to this clip, then offset by cumulative time
                        sub_start_rel = max(0, overlap_start - start) + cumulative_time
                        sub_end_rel = min(clip_duration, overlap_end - start) + cumulative_time
                        duration_seg = sub_end_rel - sub_start_rel
                        
                        if duration_seg > 0.1:  # Minimum duration for readability
                            try:
                                # Increase font size for high energy segments
                                energy_scale = {"low": 0.9, "medium": 1.0, "high": 1.1, "climax": 1.2}
                                base_size = int(subtitle_font_size * 2 * energy_scale.get(energy_level, 1.0))
                                
                                # Create animated caption clip
                                caption_clip = _create_animated_caption_clip(
                                    words=group["words"],
                                    font_path=font_path,
                                    base_font_size=base_size,
                                    duration=duration_seg,
                                    start_time=sub_start_rel,
                                    target_width=target_width,
                                    target_height=target_height,
                                    y_position=0.55  # Slightly below center
                                )
                                all_subtitle_clips.append(caption_clip)
                            except Exception as e:
                                print(f"Error creating animated caption: {e}")
                                # Fallback to simple text
                                words_text = " ".join([w["word"].upper() for w in group["words"]])
                                txt_clip = _create_text_image_clip(
                                    text=words_text,
                                    font_path=font_path,
                                    font_size=int(subtitle_font_size * 2),
                                    text_color=(255, 255, 0),
                                    stroke_color=(0, 0, 0),
                                    stroke_width=subtitle_stroke * 3,
                                    max_width=int(target_width * 0.8),
                                    duration=duration_seg,
                                    start_time=sub_start_rel,
                                    position=('center', 'center')
                                )
                                all_subtitle_clips.append(txt_clip)
                
                video_clips.append(clip)
                cumulative_time += clip_duration
        
        print(f"[Performance] Created {len(all_subtitle_clips)} animated caption clips")
        
        if video_clips:
            # Apply Transitions
            final_clips_with_effects = []
            for i in range(len(video_clips)):
                current_clip = video_clips[i]
                transition_type = cuts[i].get("transition", "cut") if i < len(cuts) else "cut"
                
                if i < len(video_clips) - 1:
                    next_clip = video_clips[i+1]
                    
                    if transition_type == "crossfade":
                        min_dur = min(current_clip.duration, next_clip.duration)
                        duration = min(1.0, min_dur / 2.0)
                        current_clip = current_clip.with_effects([vfx.FadeOut(duration)])
                        video_clips[i+1] = next_clip.with_effects([vfx.FadeIn(duration)])
                    elif transition_type == "fade_to_black":
                        duration = 0.5
                        current_clip = current_clip.with_effects([vfx.FadeOut(duration)])
                        video_clips[i+1] = next_clip.with_effects([vfx.FadeIn(duration)])
                
                final_clips_with_effects.append(current_clip)
            
            print(f"Concatenating {len(final_clips_with_effects)} clips")
            base_video = concatenate_videoclips(final_clips_with_effects, method="compose")
            
            # --- OPTIMIZATION 4: Single flat CompositeVideoClip for all overlays ---
            overlay_clips = []
            
            # Add heading overlay (also using Pillow)
            heading = state.get("heading")
            bar_height = int(target_height * 0.08)
            if heading:
                try:
                    gradient_bar = ColorClip(size=(target_width, bar_height), color=(0, 0, 0))
                    gradient_bar = gradient_bar.with_opacity(0.65).with_duration(base_video.duration)
                    gradient_bar = gradient_bar.with_position((0, 0))
                    overlay_clips.append(gradient_bar)
                    
                    # Use Pillow-based heading (FAST)
                    heading_clip = _create_text_image_clip(
                        text=heading,
                        font_path=font_path,
                        font_size=heading_font_size,
                        text_color=(255, 255, 255),  # White
                        stroke_color=(0, 0, 0),      # Black
                        stroke_width=heading_stroke,
                        max_width=int(target_width * 0.9),
                        duration=base_video.duration,
                        start_time=0,
                        position=('center', bar_height // 4)  # Vertically center in bar
                    )
                    overlay_clips.append(heading_clip)
                    print(f"Added heading overlay: {heading}")
                except Exception as e:
                    print(f"Could not add heading overlay: {e}")
                    logger.log_event("warning", {"node": "edit_video", "warning": f"Heading overlay failed: {e}", "heading": heading})

            # Combine base video + all overlays in ONE CompositeVideoClip
            if overlay_clips or all_subtitle_clips:
                final_clip = CompositeVideoClip(
                    [base_video] + overlay_clips + all_subtitle_clips,
                    size=(target_width, target_height)
                )
            else:
                final_clip = base_video

            # --- Add Background Music with Ducking ---
            # Create speech segments from cuts for audio ducking
            speech_segments = []
            cumulative = 0.0
            for cut in cuts:
                start = max(0, cut["start"])
                end = min(original_clip.duration, cut["end"])
                if end > start:
                    duration = end - start
                    speech_segments.append({
                        "start": cumulative,
                        "end": cumulative + duration
                    })
                    cumulative += duration
            
            final_clip = _add_background_music(
                final_clip,
                speech_segments=speech_segments,
                music_volume=0.25,  # 25% base volume
                enable_ducking=True
            )

            # --- OPTIMIZATION 5: GPU-accelerated FFmpeg encoding (NVENC) ---
            video_codec = _get_video_codec()
            preset, ffmpeg_params = _get_ffmpeg_params(video_codec)
            
            print(f"[Performance] Writing video with {video_codec} encoder...")
            final_clip.write_videofile(
                output_path, 
                codec=video_codec,
                audio_codec="aac",
                preset=preset,
                threads=0,  # Auto-detect optimal thread count
                ffmpeg_params=ffmpeg_params,
                logger=None
            )
            final_clip.close()

        original_clip.close()
        
        logger.log_event("node_end", {"node": "edit_video", "output": {"output_path": output_path}})
        return {"output_video_path": output_path}
    except Exception as e:
        print(f"Error editing video: {e}")
        logger.log_event("node_error", {"node": "edit_video", "error": str(e)})
        raise e


def run_critic(state: VideoState) -> dict:
    """Run critic agent to review and potentially refine cuts."""
    print("--- 🎭 Running Critic Agent ---")
    logger.log_event("node_start", {"node": "run_critic", "input": {"cuts_count": len(state["cuts"])}})
    
    if not state.get("use_critic", True):
        print("Critic agent disabled, skipping...")
        logger.log_event("node_skip", {"node": "run_critic", "reason": "disabled"})
        return {}
    
    try:
        llm = ModelFactory.get_model()
        critic = CriticAgent(llm)
        
        # Run critique on current cuts
        critique = critic.critique_cuts(
            cuts=state["cuts"],
            transcript_text=state["transcript_text"],
            transcript_segments=state["transcript_segments"]
        )
        
        if critique.approved:
            print(f"✅ Critic approved cuts (score: {critique.overall_score}/10)")
            logger.log_event("node_end", {"node": "run_critic", "approved": True, "score": critique.overall_score})
            return {}
        
        # If not approved and refinements suggested, apply them
        if critique.suggested_refinements:
            print(f"🔄 Critic suggested {len(critique.suggested_refinements)} refinements")
            refined_cuts = critic.apply_refinements(state["cuts"], critique.suggested_refinements)
            logger.log_event("node_end", {"node": "run_critic", "approved": False, "refinements": len(critique.suggested_refinements)})
            return {"cuts": refined_cuts}
        
        logger.log_event("node_end", {"node": "run_critic", "approved": False, "no_refinements": True})
        return {}
        
    except Exception as e:
        print(f"⚠️ Critic agent failed: {e}, continuing with original cuts")
        logger.log_event("node_error", {"node": "run_critic", "error": str(e)})
        return {}


# --- Graph Construction ---

workflow = StateGraph(VideoState)

workflow.add_node("extract_audio", extract_audio)
workflow.add_node("transcribe", transcribe_audio)
workflow.add_node("analyze_visually", analyze_video_visually)  # Vision LLM analysis
workflow.add_node("analyze", analyze_transcript)
workflow.add_node("run_critic", run_critic)  # Critic agent for quality review
workflow.add_node("generate_heading", generate_heading)
workflow.add_node("edit_video", edit_video)

workflow.set_entry_point("extract_audio")

# Flow: extract -> transcribe -> visual analysis -> analyze -> critic -> heading -> edit
workflow.add_edge("extract_audio", "transcribe")
workflow.add_edge("transcribe", "analyze_visually")
workflow.add_edge("analyze_visually", "analyze")
workflow.add_edge("analyze", "run_critic")
workflow.add_edge("run_critic", "generate_heading")
workflow.add_edge("generate_heading", "edit_video")
workflow.add_edge("edit_video", END)

app = workflow.compile()


# --- Main Execution ---
if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="PRISM Video Graph - Viral shorts generator")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--dev", action="store_true", help="Development mode: cache transcriptions for faster iteration")
    parser.add_argument("--preview", action="store_true", help="Preview mode: lower quality, faster export (512x720)")
    parser.add_argument("--production", action="store_true", help="Production mode: full quality export (1080x1920)")
    parser.add_argument("--no-critic", action="store_true", help="Disable critic agent review")
    parser.add_argument("--no-music", action="store_true", help="Disable background music")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file '{args.video_path}' not found.")
        sys.exit(1)
    
    # Determine export mode (default to preview for faster iteration)
    if args.production:
        export_mode = "production"
        print("🎬 [PRODUCTION MODE] Full quality export (1080x1920)")
    else:
        export_mode = "preview"
        print("⚡ [PREVIEW MODE] Fast export (512x720)")

    if args.dev:
        print("📦 [DEV MODE] Transcription caching enabled")
    
    if args.no_critic:
        print("🚫 Critic agent disabled")
    
    if args.no_music:
        print("🔇 Background music disabled")
    
    print(f"\n🎥 Processing video: {args.video_path}")
    initial_state: VideoState = cast(VideoState, {
        "input_video_path": args.video_path, 
        "dev_mode": args.dev,
        "export_mode": export_mode,
        "use_critic": not args.no_critic,
        "use_music": not args.no_music,
        "audio_path": "",
        "transcript_text": "",
        "transcript_segments": [],
        "cuts": [],
        "heading": "",
        "output_video_path": "",
        "parameters": {}
    })
    
    try:
        final_state = app.invoke(initial_state)
        print(f"\n✅ Video processing complete! Output saved to: {final_state['output_video_path']}")
        logger.log_event("run_complete", {
            "output_video_path": final_state['output_video_path'], 
            "dev_mode": args.dev,
            "export_mode": export_mode,
            "use_critic": not args.no_critic,
            "use_music": not args.no_music
        })
    except Exception as e:
        print(f"\n❌ An error occurred during execution: {e}")
        logger.log_event("run_failed", {"error": str(e)})
    finally:
        # Cleanup temp files
        if os.path.exists("temp_audio.mp3"):
            os.remove("temp_audio.mp3")
            print("🧹 Cleaned up temp_audio.mp3")
