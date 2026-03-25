"""
State Definition Module for PRISM
=================================
LangGraph state type definitions for the video processing workflow.
"""

from typing import TypedDict, List, Dict, Any


class VideoState(TypedDict):
    """State dictionary for the PRISM video processing workflow."""
    input_video_path: str
    audio_path: str
    transcript_text: str
    transcript_segments: List[Dict[str, Any]]  # List of {start: float, end: float, text: str}
    cuts: List[Dict[str, Any]]  # {start: float, end: float, reason: str, energy_level: str, is_hook: bool}
    heading: str
    output_video_path: str
    dev_mode: bool  # Development mode: cache transcriptions
    export_mode: str  # "preview" or "production" - controls output quality
    use_critic: bool  # Whether to run critic agent for quality review
    use_music: bool  # Whether to add background music
    parameters: Dict[str, Any]  # For extensibility/future use