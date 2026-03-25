"""
PRISM Music Agent - Intelligent Background Music Integration
============================================================
Adds professional background music with audio ducking for viral short-form content.

Features:
- Mood detection from transcript using LLM
- Royalty-free music selection from curated library
- Professional audio ducking during speech
- Beat-aligned transitions (future)
- Dynamic volume adjustment based on content intensity
"""

import os
import json
from typing import TypedDict, List, Dict, Any, Optional, Literal
from dataclasses import dataclass
from enum import Enum
import numpy as np

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from moviepy import AudioFileClip, CompositeAudioClip, concatenate_audioclips
from moviepy.audio.fx import AudioFadeIn, AudioFadeOut, MultiplyVolume

# Try to import audio analysis libraries
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("[MusicAgent] Warning: librosa not installed. Advanced audio analysis disabled.")


class MusicMood(str, Enum):
    """Supported music moods for content matching."""
    ENERGETIC = "energetic"      # High energy, fast-paced content
    DRAMATIC = "dramatic"        # Serious, impactful moments
    UPLIFTING = "uplifting"      # Positive, motivational content
    CHILL = "chill"              # Relaxed, conversational content
    INTENSE = "intense"          # Tension, conflict, debate
    INSPIRATIONAL = "inspirational"  # Success stories, achievements
    NEUTRAL = "neutral"          # Generic background


class MoodAnalysisResult(BaseModel):
    """Structured output for mood analysis."""
    primary_mood: str = Field(description="Primary mood of the content")
    secondary_mood: Optional[str] = Field(default=None, description="Secondary mood if mixed content")
    energy_level: int = Field(ge=1, le=10, description="Energy level from 1-10")
    reasoning: str = Field(description="Brief explanation of mood selection")


@dataclass
class MusicTrack:
    """Represents a music track in the library."""
    name: str
    path: str
    mood: MusicMood
    bpm: Optional[int] = None
    duration: Optional[float] = None
    energy_level: int = 5  # 1-10 scale


class MusicLibrary:
    """
    Curated royalty-free music library.
    
    In production, this would connect to a database or API.
    For hackathon demo, we use placeholder tracks that can be replaced.
    """
    
    # Default music directory
    MUSIC_DIR = "music_library"
    
    # Placeholder tracks - replace with actual royalty-free music files
    PLACEHOLDER_TRACKS = {
        MusicMood.ENERGETIC: [
            {"name": "high_energy_beat", "bpm": 128, "energy": 9},
            {"name": "power_up", "bpm": 140, "energy": 8},
        ],
        MusicMood.DRAMATIC: [
            {"name": "epic_cinematic", "bpm": 90, "energy": 7},
            {"name": "tension_builder", "bpm": 85, "energy": 6},
        ],
        MusicMood.UPLIFTING: [
            {"name": "feel_good_vibes", "bpm": 110, "energy": 7},
            {"name": "sunny_day", "bpm": 120, "energy": 6},
        ],
        MusicMood.CHILL: [
            {"name": "lo_fi_study", "bpm": 85, "energy": 3},
            {"name": "ambient_waves", "bpm": 75, "energy": 2},
        ],
        MusicMood.INTENSE: [
            {"name": "dark_pulse", "bpm": 130, "energy": 8},
            {"name": "confrontation", "bpm": 125, "energy": 9},
        ],
        MusicMood.INSPIRATIONAL: [
            {"name": "rise_up", "bpm": 100, "energy": 7},
            {"name": "breakthrough", "bpm": 95, "energy": 6},
        ],
        MusicMood.NEUTRAL: [
            {"name": "soft_background", "bpm": 90, "energy": 4},
            {"name": "corporate_minimal", "bpm": 100, "energy": 3},
        ],
    }
    
    def __init__(self, music_dir: Optional[str] = None):
        self.music_dir = music_dir or self.MUSIC_DIR
        self.tracks: List[MusicTrack] = []
        self._load_library()
    
    def _load_library(self):
        """Load music tracks from directory or use placeholders."""
        if os.path.exists(self.music_dir):
            # Load actual music files
            for mood in MusicMood:
                mood_dir = os.path.join(self.music_dir, mood.value)
                if os.path.exists(mood_dir):
                    for file in os.listdir(mood_dir):
                        if file.endswith(('.mp3', '.wav', '.ogg', '.m4a')):
                            self.tracks.append(MusicTrack(
                                name=os.path.splitext(file)[0],
                                path=os.path.join(mood_dir, file),
                                mood=mood
                            ))
        
        # If no tracks loaded, create placeholder entries
        if not self.tracks:
            print(f"[MusicLibrary] No music files found. Using placeholder configuration.")
            print(f"[MusicLibrary] To add music, create: {self.music_dir}/<mood>/<track>.mp3")
    
    def get_track_for_mood(self, mood: MusicMood, energy_level: int = 5) -> Optional[MusicTrack]:
        """Select best matching track for given mood and energy level."""
        matching_tracks = [t for t in self.tracks if t.mood == mood]
        
        if not matching_tracks:
            # Try neutral as fallback
            matching_tracks = [t for t in self.tracks if t.mood == MusicMood.NEUTRAL]
        
        if not matching_tracks:
            return None
        
        # Sort by energy level proximity
        matching_tracks.sort(key=lambda t: abs(t.energy_level - energy_level))
        return matching_tracks[0]


class AudioDucker:
    """
    Professional audio ducking implementation.
    
    Automatically reduces music volume during speech for clarity.
    """
    
    def __init__(
        self,
        duck_level: float = 0.15,       # Volume multiplier during speech (0-1)
        attack_time: float = 0.1,       # Fade down time in seconds
        release_time: float = 0.3,      # Fade up time in seconds
        threshold_db: float = -30,      # Speech detection threshold
        lookahead: float = 0.05         # Anticipate speech start
    ):
        self.duck_level = duck_level
        self.attack_time = attack_time
        self.release_time = release_time
        self.threshold_db = threshold_db
        self.lookahead = lookahead
    
    def create_ducking_envelope(
        self,
        speech_segments: List[Dict[str, float]],
        total_duration: float,
        sample_rate: int = 44100
    ) -> np.ndarray:
        """
        Create a volume envelope for ducking based on speech segments.
        
        Args:
            speech_segments: List of {start, end} dicts for speech timing
            total_duration: Total audio duration in seconds
            sample_rate: Audio sample rate
            
        Returns:
            numpy array of volume multipliers (0-1)
        """
        num_samples = int(total_duration * sample_rate)
        envelope = np.ones(num_samples)
        
        attack_samples = int(self.attack_time * sample_rate)
        release_samples = int(self.release_time * sample_rate)
        lookahead_samples = int(self.lookahead * sample_rate)
        
        for seg in speech_segments:
            start_sample = max(0, int(seg['start'] * sample_rate) - lookahead_samples)
            end_sample = min(num_samples, int(seg['end'] * sample_rate))
            
            # Apply duck level during speech
            envelope[start_sample:end_sample] = self.duck_level
            
            # Smooth attack (fade down)
            if start_sample > attack_samples:
                attack_start = start_sample - attack_samples
                attack_curve = np.linspace(1.0, self.duck_level, attack_samples)
                envelope[attack_start:start_sample] = np.minimum(
                    envelope[attack_start:start_sample],
                    attack_curve
                )
            
            # Smooth release (fade up)
            if end_sample + release_samples < num_samples:
                release_curve = np.linspace(self.duck_level, 1.0, release_samples)
                envelope[end_sample:end_sample + release_samples] = np.maximum(
                    envelope[end_sample:end_sample + release_samples],
                    release_curve
                )
        
        return envelope
    
    def apply_ducking(
        self,
        music_audio: AudioFileClip,
        speech_segments: List[Dict[str, float]],
        sample_rate: int = 44100
    ) -> AudioFileClip:
        """
        Apply ducking to music audio based on speech segments.
        
        Args:
            music_audio: MoviePy AudioFileClip of background music
            speech_segments: List of {start, end} timing for speech
            sample_rate: Audio sample rate
            
        Returns:
            AudioFileClip with ducking applied
        """
        envelope = self.create_ducking_envelope(
            speech_segments,
            music_audio.duration,
            sample_rate
        )
        
        def apply_envelope(get_frame):
            def new_get_frame(t):
                frame = get_frame(t)
                if isinstance(t, np.ndarray):
                    # Vectorized time array
                    indices = (t * sample_rate).astype(int)
                    indices = np.clip(indices, 0, len(envelope) - 1)
                    multipliers = envelope[indices]
                    if frame.ndim == 1:
                        return frame * multipliers
                    else:
                        return frame * multipliers[:, np.newaxis]
                else:
                    # Single time value
                    idx = min(int(t * sample_rate), len(envelope) - 1)
                    return frame * envelope[idx]
            return new_get_frame
        
        # Create new audio with ducking applied
        ducked_audio = music_audio.transform(apply_envelope)
        return ducked_audio


class MusicAgent:
    """
    AI-powered Music Agent for PRISM.
    
    Analyzes video content and applies intelligent background music
    with professional audio ducking.
    """
    
    def __init__(
        self,
        music_library: Optional[MusicLibrary] = None,
        base_music_volume: float = 0.25,  # Base volume for background music (0-1)
        duck_level: float = 0.12,         # Volume during speech
        fade_duration: float = 2.0        # Fade in/out duration
    ):
        self.library = music_library or MusicLibrary()
        self.base_volume = base_music_volume
        self.ducker = AudioDucker(duck_level=duck_level)
        self.fade_duration = fade_duration
    
    def analyze_mood(
        self,
        transcript_text: str,
        llm,
        logger=None
    ) -> MoodAnalysisResult:
        """
        Analyze transcript to determine content mood using LLM.
        
        Args:
            transcript_text: Full video transcript
            llm: LangChain LLM instance
            logger: Optional structured logger
            
        Returns:
            MoodAnalysisResult with mood and energy level
        """
        from llm_core import call_llm_with_structure
        
        system_prompt = """
You are a music supervisor for viral short-form video content.
Analyze the transcript and determine the best background music mood.

Available moods:
- energetic: High energy, fast-paced, exciting content
- dramatic: Serious, impactful, newsworthy moments  
- uplifting: Positive, motivational, feel-good content
- chill: Relaxed, conversational, casual content
- intense: Tension, conflict, debate, confrontation
- inspirational: Success stories, achievements, breakthroughs
- neutral: Generic background for mixed content

Also rate the energy level from 1-10:
- 1-3: Calm, slow-paced
- 4-6: Moderate pace
- 7-10: High energy, fast-paced

CRITICAL: Respond with ONLY valid JSON in this EXACT format:
{
    "primary_mood": "energetic",
    "secondary_mood": "intense",
    "energy_level": 8,
    "reasoning": "The speaker is delivering a passionate argument with strong emotional peaks"
}
"""
        
        user_message = f"Analyze this transcript for background music selection:\n\n{transcript_text[:3000]}"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        
        if logger:
            logger.log_event("llm_call", {
                "agent": "MusicAgent",
                "action": "analyze_mood",
                "transcript_preview": transcript_text[:200]
            })
        
        try:
            result = call_llm_with_structure(llm, messages, MoodAnalysisResult)
            
            if logger:
                logger.log_event("llm_response", {
                    "agent": "MusicAgent",
                    "result": result.dict()
                })
            
            return result
            
        except Exception as e:
            print(f"[MusicAgent] Mood analysis failed: {e}")
            # Fallback to neutral
            return MoodAnalysisResult(
                primary_mood="neutral",
                secondary_mood=None,
                energy_level=5,
                reasoning="Fallback due to analysis error"
            )
    
    def select_music(
        self,
        mood_result: MoodAnalysisResult
    ) -> Optional[MusicTrack]:
        """
        Select appropriate music track based on mood analysis.
        
        Args:
            mood_result: Result from analyze_mood()
            
        Returns:
            Selected MusicTrack or None if no suitable track found
        """
        try:
            mood = MusicMood(mood_result.primary_mood)
        except ValueError:
            mood = MusicMood.NEUTRAL
        
        track = self.library.get_track_for_mood(mood, mood_result.energy_level)
        
        if track:
            print(f"[MusicAgent] Selected track: {track.name} (mood={mood.value}, energy={mood_result.energy_level})")
        else:
            print(f"[MusicAgent] No track found for mood: {mood.value}")
        
        return track
    
    def create_music_layer(
        self,
        track: MusicTrack,
        target_duration: float,
        speech_segments: List[Dict[str, float]]
    ) -> Optional[AudioFileClip]:
        """
        Create the final music audio layer with ducking applied.
        
        Args:
            track: Selected music track
            target_duration: Duration of the video
            speech_segments: List of {start, end} for speech timing
            
        Returns:
            AudioFileClip with volume adjustment and ducking applied
        """
        if not os.path.exists(track.path):
            print(f"[MusicAgent] Track file not found: {track.path}")
            return None
        
        try:
            music = AudioFileClip(track.path)
            
            # Loop or trim to match video duration
            if music.duration < target_duration:
                # Loop the music
                loops_needed = int(np.ceil(target_duration / music.duration))
                music = concatenate_audioclips([music] * loops_needed)
            
            # Trim to exact duration
            music = music.subclipped(0, target_duration)
            
            # Apply base volume
            music = music.with_effects([MultiplyVolume(self.base_volume)])
            
            # Apply fade in/out
            fade_dur = min(self.fade_duration, target_duration / 4)
            music = music.with_effects([
                AudioFadeIn(fade_dur),
                AudioFadeOut(fade_dur)
            ])
            
            # Apply ducking based on speech segments
            if speech_segments:
                music = self.ducker.apply_ducking(music, speech_segments)
            
            return music
            
        except Exception as e:
            print(f"[MusicAgent] Error creating music layer: {e}")
            return None
    
    def mix_audio(
        self,
        original_audio: AudioFileClip,
        music_audio: AudioFileClip
    ) -> CompositeAudioClip:
        """
        Mix original audio with background music.
        
        Args:
            original_audio: Original video audio (speech)
            music_audio: Background music layer (already ducked)
            
        Returns:
            CompositeAudioClip with both tracks mixed
        """
        return CompositeAudioClip([original_audio, music_audio])


def create_demo_music_structure():
    """
    Create the music library directory structure for demo purposes.
    """
    base_dir = "music_library"
    
    for mood in MusicMood:
        mood_dir = os.path.join(base_dir, mood.value)
        os.makedirs(mood_dir, exist_ok=True)
    
    # Create README with instructions
    readme_content = """# PRISM Music Library

Place royalty-free music files in the appropriate mood folders:

- **energetic/**: High energy, fast-paced tracks (EDM, upbeat)
- **dramatic/**: Cinematic, epic, serious tracks
- **uplifting/**: Feel-good, positive, happy tracks
- **chill/**: Lo-fi, ambient, relaxed tracks
- **intense/**: Dark, tense, confrontational tracks
- **inspirational/**: Motivational, achievement-oriented tracks
- **neutral/**: Minimal, unobtrusive background tracks

## Recommended Sources (Royalty-Free):
- Pixabay Music (pixabay.com/music)
- YouTube Audio Library
- Mixkit (mixkit.co)
- Bensound (bensound.com)

## File Formats Supported:
- MP3 (recommended)
- WAV
- OGG
- M4A

## Naming Convention:
`track_name_BPM.mp3` (e.g., `epic_cinematic_90.mp3`)
"""
    
    with open(os.path.join(base_dir, "README.md"), "w") as f:
        f.write(readme_content)
    
    print(f"[MusicAgent] Created music library structure at: {base_dir}/")
    print("[MusicAgent] Add royalty-free music files to the mood folders to enable background music.")


# Integration helper for video_graph.py
def add_background_music(
    video_clip,
    transcript_segments: List[Dict],
    transcript_text: str,
    llm,
    logger=None
):
    """
    High-level function to add background music to a video clip.
    
    This is the main integration point for video_graph.py
    
    Args:
        video_clip: MoviePy VideoFileClip with audio
        transcript_segments: List of transcript segments with timing
        transcript_text: Full transcript text
        llm: LangChain LLM instance
        logger: Optional structured logger
        
    Returns:
        video_clip with background music added (or original if music unavailable)
    """
    agent = MusicAgent()
    
    # Skip if no music library
    if not agent.library.tracks:
        print("[MusicAgent] No music tracks available. Skipping background music.")
        return video_clip
    
    # Analyze mood
    mood_result = agent.analyze_mood(transcript_text, llm, logger)
    
    # Select track
    track = agent.select_music(mood_result)
    if not track:
        return video_clip
    
    # Create speech segments for ducking
    speech_segments = [
        {"start": seg["start"], "end": seg["end"]}
        for seg in transcript_segments
    ]
    
    # Create music layer
    music_audio = agent.create_music_layer(
        track,
        video_clip.duration,
        speech_segments
    )
    
    if music_audio is None:
        return video_clip
    
    # Mix with original audio
    if video_clip.audio:
        mixed_audio = agent.mix_audio(video_clip.audio, music_audio)
        video_clip = video_clip.with_audio(mixed_audio)
        print(f"[MusicAgent] Successfully added background music: {track.name}")
    
    return video_clip


if __name__ == "__main__":
    # Demo: Create music library structure
    create_demo_music_structure()
