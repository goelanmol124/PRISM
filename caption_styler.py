"""
PRISM Caption Styler - Dynamic TikTok-Style Animated Captions
=============================================================
Creates eye-catching, animated subtitles that pop like viral content.

Features:
- Word-by-word animation with emphasis detection
- Multiple style presets (TikTok, Hormozi, News, Minimal)
- Emoji integration based on sentiment
- Color coding by emotion/emphasis
- Karaoke-style highlighting
- Professional typography and effects
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple, Literal
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from pydantic import BaseModel, Field


class CaptionStyle(str, Enum):
    """Available caption style presets."""
    TIKTOK = "tiktok"           # Classic TikTok centered bold style
    HORMOZI = "hormozi"         # Alex Hormozi big word pop style
    NEWS = "news"               # Lower-third news banner style
    MINIMAL = "minimal"         # Clean, minimal subtitles
    KARAOKE = "karaoke"         # Word-by-word highlight
    IMPACT = "impact"           # High impact with effects
    GRADIENT = "gradient"       # Gradient color text


class EmphasisType(str, Enum):
    """Types of word emphasis."""
    NORMAL = "normal"
    STRONG = "strong"           # Important keywords
    QUESTION = "question"       # Question words
    EXCLAIM = "exclaim"         # Exclamatory
    NUMBER = "number"           # Statistics/numbers
    QUOTE = "quote"             # Quoted text


@dataclass
class StyleConfig:
    """Configuration for a caption style preset."""
    name: str
    font_size_multiplier: float = 1.0
    primary_color: str = "white"
    secondary_color: str = "yellow"
    emphasis_color: str = "#FFD700"  # Gold
    background_color: Optional[str] = None
    background_opacity: float = 0.0
    stroke_color: str = "black"
    stroke_width: int = 3
    font_weight: str = "bold"
    position: Tuple[str, str] = ("center", "center")
    animation: str = "pop"  # pop, slide, fade, bounce
    max_words_per_line: int = 4
    word_spacing: float = 1.2
    shadow_offset: Tuple[int, int] = (2, 2)
    shadow_color: str = "black"
    shadow_opacity: float = 0.5


# Pre-defined style configurations
STYLE_PRESETS: Dict[CaptionStyle, StyleConfig] = {
    CaptionStyle.TIKTOK: StyleConfig(
        name="TikTok Classic",
        font_size_multiplier=1.8,
        primary_color="white",
        secondary_color="#00F5FF",  # Cyan
        emphasis_color="#FFD700",
        stroke_color="black",
        stroke_width=4,
        position=("center", 0.42),  # 42% from top - avoids bottom cutoff
        animation="pop",
        max_words_per_line=3
    ),
    CaptionStyle.HORMOZI: StyleConfig(
        name="Hormozi Impact",
        font_size_multiplier=2.5,
        primary_color="yellow",
        secondary_color="white",
        emphasis_color="#FF4444",  # Red for impact
        stroke_color="black",
        stroke_width=6,
        position=("center", 0.40),  # 40% from top - gives room for large text
        animation="pop",
        max_words_per_line=2
    ),
    CaptionStyle.NEWS: StyleConfig(
        name="News Lower Third",
        font_size_multiplier=1.2,
        primary_color="white",
        background_color="#1a1a2e",
        background_opacity=0.85,
        stroke_width=0,
        position=("center", 0.85),  # Lower third
        animation="slide",
        max_words_per_line=8
    ),
    CaptionStyle.MINIMAL: StyleConfig(
        name="Minimal Clean",
        font_size_multiplier=1.0,
        primary_color="white",
        stroke_color="black",
        stroke_width=2,
        position=("center", 0.75),
        animation="fade",
        max_words_per_line=6
    ),
    CaptionStyle.KARAOKE: StyleConfig(
        name="Karaoke Highlight",
        font_size_multiplier=1.5,
        primary_color="#888888",  # Dimmed for inactive
        secondary_color="white",   # Bright for active
        emphasis_color="#00FF00",  # Green for current word
        stroke_width=3,
        position=("center", 0.45),  # Slightly above center
        animation="highlight",
        max_words_per_line=4
    ),
    CaptionStyle.IMPACT: StyleConfig(
        name="High Impact",
        font_size_multiplier=2.2,
        primary_color="white",
        secondary_color="#FF0080",  # Hot pink
        emphasis_color="#00FFFF",   # Cyan
        stroke_color="black",
        stroke_width=5,
        shadow_offset=(4, 4),
        position=("center", 0.40),  # 40% from top for large text
        animation="bounce",
        max_words_per_line=2
    ),
    CaptionStyle.GRADIENT: StyleConfig(
        name="Gradient Text",
        font_size_multiplier=1.8,
        primary_color="#FF6B6B",    # Start color
        secondary_color="#4ECDC4",  # End color
        stroke_width=3,
        position=("center", "center"),
        animation="pop",
        max_words_per_line=3
    ),
}


# Emphasis keywords for automatic detection
EMPHASIS_KEYWORDS = {
    EmphasisType.STRONG: [
        "important", "critical", "essential", "must", "need", "key",
        "powerful", "amazing", "incredible", "insane", "crazy", "massive",
        "huge", "game-changer", "secret", "truth", "reality", "fact",
        "money", "million", "billion", "success", "win", "best", "worst",
        "never", "always", "only", "first", "last", "biggest", "smallest"
    ],
    EmphasisType.NUMBER: [],  # Detected via regex
    EmphasisType.QUESTION: ["why", "how", "what", "when", "where", "who", "which"],
    EmphasisType.EXCLAIM: ["wow", "yes", "no", "stop", "wait", "look", "listen"],
}


# Emoji mappings for sentiment/context
SENTIMENT_EMOJIS = {
    "money": "💰",
    "success": "🚀",
    "warning": "⚠️",
    "important": "❗",
    "idea": "💡",
    "fire": "🔥",
    "growth": "📈",
    "down": "📉",
    "time": "⏰",
    "question": "❓",
    "love": "❤️",
    "mind": "🧠",
    "strong": "💪",
    "star": "⭐",
    "check": "✅",
    "cross": "❌",
}


class CaptionStyler:
    """
    Dynamic caption styling engine for viral video content.
    
    Creates TikTok-style animated captions with multiple style presets,
    automatic emphasis detection, and emoji integration.
    """
    
    def __init__(
        self,
        style: CaptionStyle = CaptionStyle.HORMOZI,
        target_width: int = 1080,
        target_height: int = 1920,
        base_font_size: int = 50,
        enable_emojis: bool = False,
        auto_emphasis: bool = True
    ):
        self.style = style
        self.config = STYLE_PRESETS[style]
        self.target_width = target_width
        self.target_height = target_height
        self.base_font_size = base_font_size
        self.enable_emojis = enable_emojis
        self.auto_emphasis = auto_emphasis
    
    def detect_emphasis(self, word: str) -> EmphasisType:
        """
        Detect the emphasis type for a word.
        
        Args:
            word: The word to analyze
            
        Returns:
            EmphasisType for the word
        """
        word_lower = word.lower().strip('.,!?;:')
        
        # Check for numbers/statistics
        if re.search(r'\d+', word):
            return EmphasisType.NUMBER
        
        # Check for question words
        if word_lower in EMPHASIS_KEYWORDS[EmphasisType.QUESTION]:
            return EmphasisType.QUESTION
        
        # Check for exclamatory words
        if word_lower in EMPHASIS_KEYWORDS[EmphasisType.EXCLAIM]:
            return EmphasisType.EXCLAIM
        
        # Check for strong emphasis keywords
        if word_lower in EMPHASIS_KEYWORDS[EmphasisType.STRONG]:
            return EmphasisType.STRONG
        
        # Check if word ends with exclamation
        if word.endswith('!'):
            return EmphasisType.EXCLAIM
        
        return EmphasisType.NORMAL
    
    def get_word_color(self, word: str, emphasis: EmphasisType) -> str:
        """
        Get the appropriate color for a word based on emphasis.
        
        Args:
            word: The word
            emphasis: Detected emphasis type
            
        Returns:
            Color string (hex or name)
        """
        if emphasis == EmphasisType.STRONG:
            return self.config.emphasis_color
        elif emphasis == EmphasisType.NUMBER:
            return "#00FF00"  # Green for numbers/stats
        elif emphasis == EmphasisType.EXCLAIM:
            return "#FF4444"  # Red for exclamations
        elif emphasis == EmphasisType.QUESTION:
            return "#00BFFF"  # Blue for questions
        else:
            return self.config.primary_color
    
    def get_word_scale(self, word: str, emphasis: EmphasisType) -> float:
        """
        Get the scale multiplier for a word based on emphasis.
        
        Args:
            word: The word
            emphasis: Detected emphasis type
            
        Returns:
            Scale multiplier (1.0 = normal)
        """
        base_scale = 1.0
        
        if emphasis == EmphasisType.STRONG:
            base_scale = 1.3
        elif emphasis == EmphasisType.NUMBER:
            base_scale = 1.2
        elif emphasis == EmphasisType.EXCLAIM:
            base_scale = 1.25
        elif emphasis == EmphasisType.QUESTION:
            base_scale = 1.1
        
        # Longer words slightly smaller
        if len(word) > 8:
            base_scale *= 0.9
        
        return base_scale
    
    def add_emoji(self, word: str) -> str:
        """
        Optionally add relevant emoji to a word.
        
        Args:
            word: The word to potentially enhance
            
        Returns:
            Word with emoji if applicable
        """
        if not self.enable_emojis:
            return word
        
        word_lower = word.lower().strip('.,!?;:')
        
        # Check for emoji triggers
        for trigger, emoji in SENTIMENT_EMOJIS.items():
            if trigger in word_lower:
                return f"{word} {emoji}"
        
        return word
    
    def create_styled_word_clip(
        self,
        word: str,
        start_time: float,
        duration: float,
        font_path: Optional[str] = None
    ):
        """
        Create a styled TextClip for a single word.
        
        Args:
            word: The word to display
            start_time: When the word appears
            duration: How long the word is displayed
            font_path: Path to font file
            
        Returns:
            MoviePy TextClip with styling applied
        """
        from moviepy import TextClip
        
        # Detect emphasis
        emphasis = self.detect_emphasis(word) if self.auto_emphasis else EmphasisType.NORMAL
        
        # Get styling
        color = self.get_word_color(word, emphasis)
        scale = self.get_word_scale(word, emphasis)
        
        # Calculate font size
        font_size = int(
            self.base_font_size * 
            self.config.font_size_multiplier * 
            scale
        )
        
        # Process word (add emoji, uppercase for impact)
        display_word = self.add_emoji(word)
        if self.style in [CaptionStyle.HORMOZI, CaptionStyle.IMPACT]:
            display_word = display_word.upper()
        
        # Create text clip
        try:
            txt_clip = TextClip(
                text=display_word,
                font_size=font_size,
                color=color,
                font=font_path,
                stroke_color=self.config.stroke_color,
                stroke_width=self.config.stroke_width,
                method='caption',
                size=(int(self.target_width * 0.85), None),
                text_align='center'
            )
            
            # Apply position
            position = self.config.position
            if isinstance(position[1], float):
                # Convert relative position to pixels
                y_pos = int(self.target_height * position[1])
                position = (position[0], y_pos)
            
            txt_clip = txt_clip.with_position(position)
            txt_clip = txt_clip.with_start(start_time)
            txt_clip = txt_clip.with_duration(duration)
            
            return txt_clip
            
        except Exception as e:
            print(f"[CaptionStyler] Error creating clip for '{word}': {e}")
            return None
    
    def create_caption_sequence(
        self,
        transcript_segments: List[Dict[str, Any]],
        clip_start: float,
        clip_end: float,
        font_path: Optional[str] = None
    ) -> List:
        """
        Create a sequence of styled caption clips for a video segment.
        
        Args:
            transcript_segments: List of transcript segments with word timing
            clip_start: Start time of the video clip
            clip_end: End time of the video clip
            font_path: Path to font file
            
        Returns:
            List of TextClip objects for compositing
        """
        caption_clips = []
        
        for seg in transcript_segments:
            # Get word-level timing if available
            words_data = seg.get("words", [])
            
            if not words_data:
                # Fallback: treat entire segment as one block
                words_data = [{
                    "word": seg["text"],
                    "start": seg["start"],
                    "end": seg["end"]
                }]
            
            for word_info in words_data:
                word_text = word_info.get("word", "").strip()
                if not word_text:
                    continue
                
                word_start = word_info["start"]
                word_end = word_info["end"]
                
                # Check if word overlaps with clip
                overlap_start = max(clip_start, word_start)
                overlap_end = min(clip_end, word_end)
                
                if overlap_end > overlap_start:
                    # Calculate relative timing within clip
                    rel_start = max(0, overlap_start - clip_start)
                    rel_duration = overlap_end - overlap_start
                    
                    if rel_duration > 0.05:  # Minimum display time
                        clip = self.create_styled_word_clip(
                            word_text,
                            rel_start,
                            rel_duration,
                            font_path
                        )
                        if clip:
                            caption_clips.append(clip)
        
        return caption_clips
    
    def create_karaoke_line(
        self,
        words: List[Dict[str, Any]],
        line_start: float,
        line_end: float,
        current_word_idx: int,
        font_path: Optional[str] = None
    ):
        """
        Create a karaoke-style line with highlighted current word.
        
        Args:
            words: List of words in the line
            line_start: Start time of the line
            line_end: End time of the line
            current_word_idx: Index of currently spoken word
            font_path: Path to font file
            
        Returns:
            CompositeVideoClip with karaoke styling
        """
        from moviepy import TextClip, CompositeVideoClip
        
        clips = []
        x_offset = 0
        
        font_size = int(
            self.base_font_size * 
            self.config.font_size_multiplier
        )
        
        for i, word_info in enumerate(words):
            word = word_info.get("word", "").strip()
            if not word:
                continue
            
            # Highlight current word
            if i == current_word_idx:
                color = self.config.emphasis_color
                scale = 1.1
            elif i < current_word_idx:
                color = self.config.secondary_color  # Already spoken
                scale = 1.0
            else:
                color = self.config.primary_color  # Not yet spoken
                scale = 1.0
            
            try:
                txt = TextClip(
                    text=word + " ",
                    font_size=int(font_size * scale),
                    color=color,
                    font=font_path,
                    stroke_color=self.config.stroke_color,
                    stroke_width=self.config.stroke_width
                )
                
                # Position horizontally
                txt = txt.with_position((x_offset, self.target_height // 2))
                clips.append(txt)
                
                x_offset += txt.w
                
            except Exception as e:
                print(f"[CaptionStyler] Karaoke error for '{word}': {e}")
        
        if clips:
            # Center the entire line
            total_width = x_offset
            offset = (self.target_width - total_width) // 2
            
            for clip in clips:
                pos = clip.pos(0)
                clip = clip.with_position((pos[0] + offset, pos[1]))
            
            return CompositeVideoClip(clips, size=(self.target_width, self.target_height))
        
        return None


def get_style_preview() -> str:
    """
    Generate a preview of all available caption styles.
    
    Returns:
        Formatted string showing all styles and their characteristics
    """
    preview = "🎨 PRISM Caption Styles\n" + "=" * 40 + "\n\n"
    
    for style, config in STYLE_PRESETS.items():
        preview += f"📌 {style.value.upper()}: {config.name}\n"
        preview += f"   Font Size: {config.font_size_multiplier}x\n"
        preview += f"   Colors: Primary={config.primary_color}, Emphasis={config.emphasis_color}\n"
        preview += f"   Animation: {config.animation}\n"
        preview += f"   Position: {config.position}\n"
        preview += "\n"
    
    return preview


# Integration helper for video_graph.py
def create_styled_captions(
    transcript_segments: List[Dict],
    clip_start: float,
    clip_end: float,
    style: CaptionStyle = CaptionStyle.HORMOZI,
    font_path: Optional[str] = None,
    target_width: int = 1080,
    target_height: int = 1920,
    enable_emojis: bool = False
) -> List:
    """
    High-level function to create styled captions for a video clip.
    
    This is the main integration point for video_graph.py
    
    Args:
        transcript_segments: List of transcript segments with word timing
        clip_start: Start time of the clip being processed
        clip_end: End time of the clip being processed
        style: Caption style preset to use
        font_path: Path to font file
        target_width: Output video width
        target_height: Output video height
        enable_emojis: Whether to add context emojis
        
    Returns:
        List of TextClip objects ready for compositing
    """
    styler = CaptionStyler(
        style=style,
        target_width=target_width,
        target_height=target_height,
        enable_emojis=enable_emojis
    )
    
    return styler.create_caption_sequence(
        transcript_segments,
        clip_start,
        clip_end,
        font_path
    )


if __name__ == "__main__":
    # Demo: Print style preview
    print(get_style_preview())
