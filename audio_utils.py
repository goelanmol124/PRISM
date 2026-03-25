"""
Audio Utilities Module for PRISM
================================
Background music and audio processing for engaging short-form content.

Features:
- Background music selection and management
- Audio ducking during speech segments
- Audio mixing and volume control
- Smart track looping and duration matching
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
from moviepy import AudioFileClip, CompositeAudioClip, AudioClip, concatenate_audioclips

# Default music directory
DEFAULT_MUSIC_DIR = "assets/music"


def get_available_music_tracks(music_dir: str = DEFAULT_MUSIC_DIR) -> List[str]:
    """Get list of available background music files."""
    if not os.path.exists(music_dir):
        os.makedirs(music_dir, exist_ok=True)
        return []
    
    music_extensions = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
    tracks = []
    for f in os.listdir(music_dir):
        if os.path.splitext(f)[1].lower() in music_extensions:
            tracks.append(os.path.join(music_dir, f))
    return tracks


def select_music_for_energy(
    energy_levels: List[str],
    music_dir: str = DEFAULT_MUSIC_DIR
) -> Optional[str]:
    """
    Select appropriate background music based on video energy levels.
    Returns path to music file or None if no music available.
    
    For now, just returns the first available track.
    Future: match music BPM/mood to video energy.
    """
    tracks = get_available_music_tracks(music_dir)
    if not tracks:
        return None
    
    # Simple selection - use first track
    # TODO: Implement mood matching based on energy_levels
    return tracks[0]


def create_ducking_envelope(
    speech_segments: List[Dict[str, float]],
    total_duration: float,
    sample_rate: int = 44100,
    duck_level: float = 0.25,      # Volume during speech (0.25 = 25%)
    attack_time: float = 0.1,       # Fade down time
    release_time: float = 0.3      # Fade up time
) -> np.ndarray:
    """
    Create a volume envelope array for ducking music under speech.
    
    Returns numpy array of volume multipliers (0-1) for each sample.
    """
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


def mix_audio_tracks(
    original_audio: Optional[AudioClip],
    music_audio: Optional[AudioClip],
    music_volume: float = 0.3  # Base music volume (before ducking)
) -> Optional[AudioClip]:
    """
    Mix original video audio with background music.
    Returns combined AudioClip.
    """
    if original_audio is None:
        return music_audio.with_volume_scaled(music_volume) if music_audio else None
    
    if music_audio is None:
        return original_audio
    
    # Scale music volume
    music_scaled = music_audio.with_volume_scaled(music_volume)
    
    # Composite the audio tracks
    return CompositeAudioClip([original_audio, music_scaled])


def add_background_music(
    video_clip,
    speech_segments: List[Dict[str, float]],
    music_path: Optional[str] = None,
    music_volume: float = 0.3,
    enable_ducking: bool = True,
    music_dir: str = DEFAULT_MUSIC_DIR
):
    """
    Add background music to video with automatic ducking during speech.
    
    Args:
        video_clip: The video clip to add music to
        speech_segments: List of {start, end} dicts for speech timing
        music_path: Path to music file (or None to auto-select)
        music_volume: Base volume for music (0-1)
        enable_ducking: Whether to duck music during speech
        music_dir: Directory to search for music files
    
    Returns:
        Video clip with background music added
    """
    # Select music track
    if music_path is None:
        music_path = select_music_for_energy([], music_dir)
    
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
            music_clip = concatenate_audioclips([music_clip] * loops_needed)
        
        # Trim to video duration
        music_clip = music_clip.subclipped(0, video_duration)
        
        # Apply ducking if enabled
        if enable_ducking and speech_segments:
            print(f"[Music] Applying audio ducking ({len(speech_segments)} speech segments)")
            envelope = create_ducking_envelope(
                speech_segments=speech_segments,
                total_duration=video_duration,
                duck_level=0.2,  # Duck to 20% during speech
                attack_time=0.15,
                release_time=0.4
            )
            
            # Scale music volume
            music_audio = music_clip.with_volume_scaled(music_volume)
            
            # Create ducked version using numpy
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