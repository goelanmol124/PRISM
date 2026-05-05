"""
PRISM Video Graph - Main Orchestration
======================================
AI-powered video editing pipeline that transforms long-form videos 
into short-form content for TikTok, Instagram Reels, and YouTube Shorts.

This is the main entry point that orchestrates all modules:
- caption_renderer: Pillow-based text rendering
- video_effects: Zoom and speed effects  
- video_utils: Cropping, encoding, face detection
- audio_utils: Background music and ducking
- vision_analysis: Keyframe extraction and vision LLM
- clip_selector: LLM-based viral moment detection
- critic_agent: Quality review and refinement
"""

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
import argparse

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from moviepy import VideoFileClip, concatenate_videoclips
import whisper
import torch

# Import our modular components
from llm_core import call_llm_with_structure, AnalysisResult, HeadingResult, CaptionAnalysis, CaptionGroup, CriticResult
from model_factory import ModelFactory
from critic_agent import CriticAgent
from state import VideoState
from logger import StructuredLogger
from caption_renderer import (
    resolve_font, create_text_image_clip, create_animated_caption_clip,
    group_words_for_captions
)
from video_effects import apply_zoom_effect, apply_speed_ramp
from video_utils import (
    get_video_codec, get_ffmpeg_params, get_face_cascade, detect_face_offset,
    crop_to_vertical, get_relevant_words_for_cuts, get_video_hash
)
from audio_utils import add_background_music
from vision_analysis import extract_keyframes, analyze_keyframes_with_vision
from clip_selector import analyze_transcript_for_cuts, generate_contextual_heading

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

# Global logger instance
logger = StructuredLogger()

# --- Graph Nodes ---

def extract_audio(state: VideoState) -> Dict[str, Any]:
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
        
        logger.log_event("node_end", {"node": "extract_audio", "output": {"audio_path": audio_path}})
        return {"audio_path": audio_path}
    except Exception as e:
        print(f"Error extracting audio: {e}")
        logger.log_event("node_error", {"node": "extract_audio", "error": str(e)})
        raise e


def transcribe_audio(state: VideoState) -> Dict[str, Any]:
    """Transcribes audio using local Whisper model with caching."""
    print("--- Transcribing Audio ---")
    logger.log_event("node_start", {"node": "transcribe_audio"})
    
    audio_path = state["audio_path"]
    video_path = state["input_video_path"]
    dev_mode = state.get("dev_mode", False)
    
    # Cache handling for dev mode
    if dev_mode:
        os.makedirs(CACHE_DIR, exist_ok=True)
        video_hash = get_video_hash(video_path)
        cache_file = os.path.join(CACHE_DIR, f"transcript_{video_hash}.json")
        
        if os.path.exists(cache_file):
            print("[DEV MODE] Loading cached transcript")
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            logger.log_event("cache_hit", {"node": "transcribe_audio", "cache_file": cache_file})
            return {
                "transcript_text": cached_data["transcript_text"],
                "transcript_segments": cached_data["transcript_segments"]
            }
    
    try:
        # Load Whisper model
        model = whisper.load_model("base")
        
        # Transcribe with word-level timestamps
        result = model.transcribe(
            audio_path,
            word_timestamps=True,
            language="en"
        )
        
        transcript_text = result["text"]
        transcript_segments = result.get("segments", [])
        
        # Cache if in dev mode
        if dev_mode:
            cache_data = {
                "transcript_text": transcript_text,
                "transcript_segments": transcript_segments
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            logger.log_event("cache_save", {"node": "transcribe_audio", "cache_file": cache_file})
        
        print(f"Transcription complete: {len(transcript_text)} characters")
        logger.log_event("node_end", {"node": "transcribe_audio", "output": {"text_length": len(transcript_text), "num_segments": len(transcript_segments)}})
        
        return {
            "transcript_text": transcript_text,
            "transcript_segments": transcript_segments
        }
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        logger.log_event("node_error", {"node": "transcribe_audio", "error": str(e)})
        raise e


def analyze_video_visually(state: VideoState) -> Dict[str, Any]:
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
    frame_paths = extract_keyframes(video_path, num_frames=5, cache_dir=CACHE_DIR)
    
    if not frame_paths:
        print("[Vision] No keyframes extracted, skipping vision analysis")
        return {"parameters": {**state.get("parameters", {}), "vision_analysis": None}}
    
    # Analyze with vision LLM
    analysis = analyze_keyframes_with_vision(frame_paths, transcript, logger)
    
    # Store in parameters for later use
    updated_params = {**state.get("parameters", {}), "vision_analysis": analysis}
    
    logger.log_event("node_end", {"node": "analyze_visually", "output": analysis})
    
    return {"parameters": updated_params}


def analyze_transcript(state: VideoState) -> Dict[str, Any]:
    """Analyzes transcript using LLM to find viral cuts."""
    cuts = analyze_transcript_for_cuts(
        state["transcript_text"],
        state["transcript_segments"],
        logger
    )
    return {"cuts": cuts}


def run_critic(state: VideoState) -> Dict[str, Any]:
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


def generate_heading(state: VideoState) -> Dict[str, Any]:
    """Generates a contextual heading for the video using LLM."""
    heading = generate_contextual_heading(state["transcript_text"], logger)
    return {"heading": heading}


def edit_video(state: VideoState) -> Dict[str, Any]:
    """Cuts and stitches 9:16 video with captions, transitions, letterboxing, and a top-bar heading."""
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

    # Top/bottom letterbox height (fraction of content height; room for heading in top bar)
    letterbox_bar = max(1, int(target_height * 0.15))

    # MoviePy imports
    from moviepy.video import fx as vfx
    from moviepy import CompositeVideoClip, VideoFileClip
    import numpy as np

    try:
        original_clip = VideoFileClip(video_path)
        
        font_path = resolve_font()

        # --- Font sizes proportional to target output width ---
        subtitle_font_size = max(16, int(target_width * 0.035))   # ~38px on 1080w
        heading_font_size = max(18, int(target_width * 0.04))     # ~43px on 1080w
        subtitle_stroke = max(1, int(target_width * 0.002))       # ~2px
        heading_stroke = max(1, int(target_width * 0.0025))       # ~3px

        # --- OPTIMIZATION 1: Pre-filter transcript to only relevant words ---
        print("[Performance] Pre-filtering transcript segments...")
        relevant_words = get_relevant_words_for_cuts(state["transcript_segments"], cuts)
        print(f"[Performance] Filtered to {len(relevant_words)} relevant words (from full transcript)")

        # --- OPTIMIZATION 2: Detect face ONCE on the original video ---
        src_w, src_h = original_clip.w, original_clip.h
        src_aspect = src_w / src_h
        face_x_offset = None
        if src_aspect > target_aspect:
            new_w = int(src_h * target_aspect)
            # Run face detection once on a sample from the middle of the video
            sample_clip = original_clip.subclipped(original_clip.duration * 0.4, min(original_clip.duration * 0.6, original_clip.duration))
            face_x_offset = detect_face_offset(sample_clip, new_w, src_w)
            if face_x_offset is None:
                face_x_offset = (src_w - new_w) // 2  # Fall back to center
                print("[Performance] Using center crop (no face detected)")
            else:
                print(f"[Performance] Face detected once, reusing offset={face_x_offset} for all cuts")

        # --- OPTIMIZATION 3: Create ANIMATED CAPTIONS with multi-word groups ---
        all_subtitle_clips = []
        video_clips = []
        cumulative_time = 0.0  # Track position in final video timeline
        
        # Group words for display (2-4 words per group for readability)
        caption_groups = group_words_for_captions(relevant_words, words_per_group=3)
        print(f"[Captions] Created {len(caption_groups)} caption groups from {len(relevant_words)} words")
        
        print("[Performance] Rendering ANIMATED captions with Pillow...")
        for cut in cuts:
            start = max(0, cut["start"])
            end = min(original_clip.duration, cut["end"])
            energy_level = cut.get("energy_level", "medium")
            is_hook = cut.get("is_hook", False)
            
            if end > start:
                clip = original_clip.subclipped(start, end)
                clip = crop_to_vertical(clip, start, face_x_offset, target_width, target_height)
                
                # --- Apply Zoom & Ken Burns Effects based on energy ---
                clip = apply_zoom_effect(clip, energy_level=energy_level, is_hook=is_hook)
                
                # --- Apply Speed Ramping for dramatic moments ---
                clip = apply_speed_ramp(clip, energy_level=energy_level)
                
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
                                base_size = int(subtitle_font_size * energy_scale.get(energy_level, 1.0))
                                
                                # Create animated caption clip
                                caption_clip = create_animated_caption_clip(
                                    words=group["words"],
                                    font_path=font_path,
                                    base_font_size=base_size,
                                    duration=duration_seg,
                                    start_time=sub_start_rel,
                                    target_width=target_width,
                                    target_height=target_height,
                                    y_position=0.55,  # Slightly below center (content area)
                                    vertical_offset=letterbox_bar,
                                )
                                all_subtitle_clips.append(caption_clip)
                            except Exception as e:
                                print(f"Error creating animated caption: {e}")
                                # Fallback to simple text
                                words_text = " ".join([w["word"].upper() for w in group["words"]])
                                txt_clip = create_text_image_clip(
                                    text=words_text,
                                    font_path=font_path,
                                    font_size=int(subtitle_font_size * 2),
                                    text_color=(255, 255, 0),
                                    stroke_color=(0, 0, 0),
                                    stroke_width=subtitle_stroke * 3,
                                    max_width=int(target_width * 0.8),
                                    duration=duration_seg,
                                    start_time=sub_start_rel,
                                    position=(
                                        'center',
                                        letterbox_bar + int(target_height * 0.55),
                                    ),
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
            
            # --- Add background music if enabled ---
            if state.get("use_music", True):
                try:
                    speech_segments = [{"start": seg["start"], "end": seg["end"]} for seg in state["transcript_segments"]]
                    base_video = add_background_music(
                        base_video,
                        speech_segments,
                        music_volume=0.25,
                        enable_ducking=True
                    )
                except Exception as e:
                    print(f"[Music] Error adding background music: {e}")
            
            # Letterboxing (solid top/bottom bars, MoviePy 2.x) — heading sits in top bar
            base_video = base_video.with_effects(
                [vfx.Margin(top=letterbox_bar, bottom=letterbox_bar, color=(0, 0, 0))]
            )
            composite_height = int(base_video.h)

            # --- OPTIMIZATION 4: Single flat CompositeVideoClip for all overlays ---
            overlay_clips: List[Any] = []

            heading = state.get("heading")
            if heading:
                try:
                    heading_clip = create_text_image_clip(
                        text=heading,
                        font_path=font_path,
                        font_size=heading_font_size,
                        text_color=(255, 255, 255),
                        stroke_color=(0, 0, 0),
                        stroke_width=heading_stroke,
                        max_width=int(target_width * 0.9),
                        duration=base_video.duration,
                        start_time=0,
                        position=('center', letterbox_bar // 4),
                    )
                    overlay_clips.append(heading_clip)
                    print(f"Added heading overlay: {heading}")
                except Exception as e:
                    print(f"Could not add heading overlay: {e}")
                    logger.log_event(
                        "warning",
                        {
                            "node": "edit_video",
                            "warning": f"Heading overlay failed: {e}",
                            "heading": heading,
                        },
                    )

            if overlay_clips or all_subtitle_clips:
                final_clip = CompositeVideoClip(
                    [base_video] + overlay_clips + all_subtitle_clips,
                    size=(target_width, composite_height),
                )
            else:
                final_clip = base_video

            # Export video
            video_codec = get_video_codec()
            preset, ffmpeg_params = get_ffmpeg_params(video_codec)
            
            print(f"[Encoding] Using {video_codec} with preset={preset}")
            
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


# --- Graph Construction ---

workflow = StateGraph(VideoState)

workflow.add_node("extract_audio", extract_audio)
workflow.add_node("transcribe", transcribe_audio)
workflow.add_node("analyze_visually", analyze_video_visually)
workflow.add_node("analyze", analyze_transcript)
workflow.add_node("run_critic", run_critic)
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