"""
Vision Analysis Module for PRISM
================================
Vision LLM integration for analyzing video content visually.

Features:
- Keyframe extraction from videos
- Vision LLM analysis of visual elements
- Scene description and mood detection
- Visual hook identification
"""

import os
import json
import base64
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
from moviepy import VideoFileClip
from langchain_core.messages import SystemMessage, HumanMessage

from model_factory import ModelFactory

DEFAULT_CACHE_DIR = ".dev_cache"


def extract_keyframes(
    video_path: str,
    num_frames: int = 5,
    uniform: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR
) -> List[str]:
    """
    Extract keyframes from video and save as temporary images.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        uniform: If True, extract uniformly distributed frames
        cache_dir: Directory to cache keyframes
    
    Returns:
        List of paths to extracted frame images
    """
    frame_paths: List[str] = []
    keyframes_dir = os.path.join(cache_dir, "keyframes")
    os.makedirs(keyframes_dir, exist_ok=True)
    
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
            
            frame_path = os.path.join(keyframes_dir, f"keyframe_{i}.jpg")
            img.save(frame_path, "JPEG", quality=85)
            frame_paths.append(frame_path)
        
        clip.close()
        print(f"[Vision] Extracted {len(frame_paths)} keyframes")
        return frame_paths
        
    except Exception as e:
        print(f"[Vision] Error extracting keyframes: {e}")
        return []


def encode_image_base64(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_keyframes_with_vision(
    frame_paths: List[str],
    transcript_preview: str = "",
    logger = None
) -> Dict[str, Any]:
    """
    Analyze keyframes using a vision LLM to understand visual context.
    
    Returns dict with:
        - scene_description: Overall description of the video
        - visual_elements: Key visual elements detected
        - visual_hooks: Visual moments that could be hooks
        - mood: Detected mood/tone
        - has_broll: Whether B-roll footage is present
        - speaker_visible: Whether a speaker is visible
        - suggested_emoji: Relevant emojis
        - color_mood: Color palette mood
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
            b64 = encode_image_base64(path)
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
        
        if logger:
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
        
        if logger:
            logger.log_event("llm_response", {
                "node": "vision_analysis",
                "result": result
            })
        
        print(f"[Vision] Analysis complete: {result.get('mood', 'unknown')} mood, "
              f"{len(result.get('visual_elements', []))} visual elements detected")
        
        return result
        
    except Exception as e:
        print(f"[Vision] Error analyzing keyframes: {e}")
        if logger:
            logger.log_event("node_error", {"node": "vision_analysis", "error": str(e)})
        return {"error": str(e)}