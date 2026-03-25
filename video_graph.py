import os
import sys
import platform
import json
import warnings
import hashlib
from typing import TypedDict, List, Any, Dict
import datetime
import uuid

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from moviepy import VideoFileClip, concatenate_videoclips, ColorClip
import whisper
import torch
from llm_core import call_llm_with_structure, AnalysisResult, HeadingResult
from model_factory import ModelFactory
from caption_styler import CaptionStyler, CaptionStyle, create_styled_captions
from critic_agent import CriticAgent, create_critic_agent

# Optional music agent import
try:
    from music_agent import MusicAgent, add_background_music
    MUSIC_AGENT_AVAILABLE = True
except ImportError:
    MUSIC_AGENT_AVAILABLE = False
    print("[PRISM] Music agent not available. Background music disabled.")

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
# Production resolution (720p vertical)
PRODUCTION_WIDTH = 720
PRODUCTION_HEIGHT = 1280

# Preview resolution (480p vertical) - for fast iteration
PREVIEW_WIDTH = 480
PREVIEW_HEIGHT = 854

# Export mode: "preview" for fast iteration, "production" for final quality
EXPORT_MODE = "preview"  # Change to "production" for final export

# Set resolution based on mode
if EXPORT_MODE == "preview":
    TARGET_WIDTH = PREVIEW_WIDTH
    TARGET_HEIGHT = PREVIEW_HEIGHT
else:
    TARGET_WIDTH = PRODUCTION_WIDTH
    TARGET_HEIGHT = PRODUCTION_HEIGHT

TARGET_ASPECT = TARGET_WIDTH / TARGET_HEIGHT  # 9:16 = 0.5625

# --- Encoding Settings ---
# Preview mode: ultra-fast, low quality
# Production mode: balanced quality/speed
ENCODING_PRESETS = {
    "preview": {
        "codec": "libx264",
        "preset": "ultrafast",
        "crf": "32",  # Lower quality, much faster
        "fps": 20,
        "threads": 16,
        "bitrate": None,
        "audio_bitrate": "64k",
    },
    "production": {
        "codec": "libx264",  # Will try VAAPI first
        "preset": "fast",
        "crf": "23",
        "fps": 30,
        "threads": 16,
        "bitrate": None,
        "audio_bitrate": "128k",
    }
}

# Hardware acceleration settings
USE_HARDWARE_ENCODING = True  # Try Intel VAAPI first
VAAPI_DEVICE = "/dev/dri/renderD128"

# --- Caption Style Configuration ---
# Change this to use different caption styles: TIKTOK, HORMOZI, NEWS, MINIMAL, KARAOKE, IMPACT, GRADIENT
CAPTION_STYLE = CaptionStyle.HORMOZI
ENABLE_BACKGROUND_MUSIC = True  # Set to False to disable music
ENABLE_EMOJIS = False  # Set to True for emoji-enhanced captions

# --- Critic Agent Configuration ---
ENABLE_CRITIC = True  # Set to False to disable narrative coherence checking
CRITIC_MAX_RETRIES = 3  # Maximum number of re-analysis attempts
CRITIC_APPROVAL_THRESHOLD = 7  # Minimum coherence score (1-10) to approve

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
    parameters: Dict[str, Any] # For extensibility/future use

# --- Helper Functions ---

def get_video_hash(video_path: str) -> str:
    """Calculate MD5 hash of video file for cache identification."""
    hash_md5 = hashlib.md5()
    with open(video_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()[:16]


def _check_vaapi_available() -> bool:
    """Check if Intel VAAPI hardware encoding is available."""
    import subprocess
    try:
        # Check if VAAPI device exists
        if not os.path.exists(VAAPI_DEVICE):
            return False
        
        # Quick test: try to initialize VAAPI encoder
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-init_hw_device", f"vaapi=va:{VAAPI_DEVICE}",
             "-f", "lavfi", "-i", "nullsrc=s=64x64:d=0.1", "-vf", "format=nv12,hwupload",
             "-c:v", "h264_vaapi", "-f", "null", "-"],
            capture_output=True, timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def _encode_video_fast(clip, output_path: str, mode: str = "preview") -> bool:
    """
    Encode video with optimized settings. Tries VAAPI hardware encoding first,
    falls back to optimized libx264 if hardware encoding fails.
    
    Args:
        clip: MoviePy video clip to encode
        output_path: Output file path
        mode: "preview" for fast/low quality, "production" for balanced
        
    Returns:
        True if encoding succeeded
    """
    import subprocess
    import tempfile
    
    settings = ENCODING_PRESETS.get(mode, ENCODING_PRESETS["preview"])
    
    # Try VAAPI hardware encoding first (production mode only, and if enabled)
    use_vaapi = (
        USE_HARDWARE_ENCODING and 
        mode == "production" and 
        _check_vaapi_available()
    )
    
    if use_vaapi:
        print(f"   🚀 Using Intel VAAPI hardware encoding")
        try:
            # For VAAPI, we need to write frames to pipe and use ffmpeg directly
            # MoviePy doesn't natively support VAAPI, so we use ffmpeg_params workaround
            
            # Write to temp file first, then re-encode with VAAPI
            temp_output = output_path + ".temp.mp4"
            clip.write_videofile(
                temp_output,
                codec="libx264",
                preset="ultrafast",  # Fast first pass
                fps=settings["fps"],
                threads=settings["threads"],
                audio_codec="aac",
                audio_bitrate=settings["audio_bitrate"],
                ffmpeg_params=["-crf", "28"],  # Reasonable quality for re-encode
                logger="bar"
            )
            
            # Re-encode with VAAPI (much faster than libx264)
            print(f"   🔄 Re-encoding with VAAPI hardware acceleration...")
            vaapi_cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
                "-init_hw_device", f"vaapi=va:{VAAPI_DEVICE}",
                "-i", temp_output,
                "-vf", "format=nv12,hwupload",
                "-c:v", "h264_vaapi",
                "-qp", "24",  # Quality parameter for VAAPI
                "-c:a", "copy",  # Copy audio (already encoded)
                output_path
            ]
            result = subprocess.run(vaapi_cmd, capture_output=True)
            
            # Clean up temp file
            if os.path.exists(temp_output):
                os.remove(temp_output)
            
            if result.returncode == 0:
                return True
            else:
                print(f"   ⚠️  VAAPI encoding failed, falling back to CPU")
                print(f"   Error: {result.stderr.decode()[:200]}")
                
        except Exception as e:
            print(f"   ⚠️  VAAPI encoding error: {e}, falling back to CPU")
            if os.path.exists(output_path + ".temp.mp4"):
                os.remove(output_path + ".temp.mp4")
    
    # Fallback: Optimized libx264 CPU encoding
    print(f"   💻 Using optimized CPU encoding (libx264)")
    
    ffmpeg_params = ["-crf", settings["crf"]]
    
    # Add tune for faster encoding
    if mode == "preview":
        ffmpeg_params.extend(["-tune", "fastdecode"])
    
    clip.write_videofile(
        output_path,
        codec=settings["codec"],
        preset=settings["preset"],
        fps=settings["fps"],
        threads=settings["threads"],
        audio_codec="aac",
        audio_bitrate=settings["audio_bitrate"],
        ffmpeg_params=ffmpeg_params,
        logger="bar"
    )
    
    return True


def _get_transcript_for_timerange(segments: List[dict], start: float, end: float) -> str:
    """
    Extract transcript text for a given time range.
    
    Args:
        segments: List of transcript segments with timing
        start: Start time in seconds
        end: End time in seconds
        
    Returns:
        Concatenated transcript text for the time range
    """
    text_parts = []
    for seg in segments:
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)
        
        # Check for overlap
        if seg_start < end and seg_end > start:
            text_parts.append(seg.get("text", "").strip())
    
    return " ".join(text_parts)

# --- Nodes ---

def extract_audio(state: VideoState):
    """Extracts audio from the input video."""
    print(f"\n{'='*60}")
    print(f"🔊 EXTRACTING AUDIO")
    print(f"{'='*60}")
    print(f"   Input: {state['input_video_path']}")
    
    logger.log_event("node_start", {"node": "extract_audio", "input": {"video_path": state["input_video_path"]}})
    
    video_path = state["input_video_path"]
    audio_path = "temp_audio.mp3"
    
    try:
        video = VideoFileClip(video_path)
        print(f"   Video duration: {video.duration:.1f}s")
        video.audio.write_audiofile(audio_path, logger=None)
        video.close()
        
        print(f"   Output: {audio_path}")
        print(f"{'='*60}\n")
        
        result = {"audio_path": audio_path}
        logger.log_event("node_end", {"node": "extract_audio", "output": result})
        return result
    except Exception as e:
        print(f"   ❌ Error: {e}")
        logger.log_event("node_error", {"node": "extract_audio", "error": str(e)})
        return {"audio_path": None} 

def transcribe_audio(state: VideoState):
    """Transcribes audio using local Whisper model. Always caches transcripts by filename."""
    print(f"\n{'='*60}")
    print(f"🎤 TRANSCRIBING AUDIO (Whisper)")
    print(f"{'='*60}")
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
        print(f"   📂 Loading cached transcript: {cache_file}")
        logger.log_event("cache_hit", {"node": "transcribe_audio", "cache_file": cache_file})
        with open(cache_file, "r", encoding="utf-8") as f:
            cached = json.load(f)
        
        print(f"   Segments: {len(cached['transcript_segments'])}")
        print(f"   Preview: \"{cached['transcript_text'][:80]}...\"")
        print(f"{'='*60}\n")
        
        logger.log_event("node_end", {"node": "transcribe_audio", "output": {"transcript_text_preview": cached["transcript_text"][:100], "segment_count": len(cached["transcript_segments"]), "from_cache": True}})
        return cached

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")
    print(f"   Model: whisper-base")
    print(f"   Processing...")
    
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
        
        print(f"   ✅ Transcription complete!")
        print(f"   Segments: {len(result['segments'])}")
        print(f"   Preview: \"{result['text'][:80]}...\"")
        print(f"   Cached to: {cache_file}")
        print(f"{'='*60}\n")
        
        logger.log_event("cache_save", {"node": "transcribe_audio", "cache_file": cache_file})
        logger.log_event("node_end", {"node": "transcribe_audio", "output": {"transcript_text_preview": result["text"][:100], "segment_count": len(result["segments"])}})
        return output
    except Exception as e:
        print(f"   ❌ Error transcribing audio: {e}")
        logger.log_event("node_error", {"node": "transcribe_audio", "error": str(e)})
        raise e

def analyze_transcript_with_feedback(
    transcript_text: str,
    segments: List[dict],
    critic_feedback: str = None,
    attempt: int = 1
) -> List[dict]:
    """
    Analyzes transcript using LLM to find viral cuts.
    Optionally incorporates critic feedback for improved selection.
    
    Args:
        transcript_text: Full transcript text
        segments: List of transcript segments with timing
        critic_feedback: Optional feedback from critic agent for retry
        attempt: Current attempt number (1-3)
    
    Returns:
        List of cut dictionaries
    """
    print(f"\n{'='*60}")
    print(f"📝 ANALYZING TRANSCRIPT (Attempt {attempt}/3)")
    print(f"{'='*60}")
    print(f"   Total segments: {len(segments)}")
    print(f"   Transcript length: {len(transcript_text)} chars")
    if critic_feedback:
        print(f"   ⚠️  Incorporating critic feedback from previous attempt")
    
    logger.log_event("node_start", {
        "node": "analyze_transcript",
        "attempt": attempt,
        "has_critic_feedback": critic_feedback is not None
    })
    
    # Initialize LLM via ModelFactory
    llm = ModelFactory.get_model(
        provider=os.getenv("LLM_PROVIDER", "openrouter"),
        model_name=os.getenv("LLM_MODEL", "meta-llama/llama-3.2-3b-instruct:free"),
        temperature=0.7
    )
    
    # Detailed Context for the LLM
    segment_details = "\n".join([f"[{s['start']:.2f}-{s['end']:.2f}]: {s['text']}" for s in segments])
    
    system_prompt = """
You are an expert video editor creating viral shorts for Gen Z.
Select the most engaging segments from the transcript to create a SHORT video.

⚠️ STRICT DURATION LIMIT: The TOTAL duration of ALL clips combined MUST be 25-30 seconds MAX.
   - Add up (end - start) for each clip
   - Total MUST NOT exceed 30 seconds
   - Aim for 25-30 seconds total

CRITICAL REQUIREMENTS:
1. TOTAL DURATION: 25-30 seconds maximum (this is NON-NEGOTIABLE)
2. Selected clips MUST form a COHERENT NARRATIVE - they should tell a complete story
3. Clips should have a clear beginning, middle, and end
4. The viewer should understand the context and message without confusion
5. Avoid jumping between unrelated topics
6. Select 2-4 clips that work together (fewer clips = more coherent)

For each cut, specify transition type to the NEXT clip:
- "cut": Fast-paced (use for 80% of transitions)
- "crossfade": Smooth flow between topics
- "fade_to_black": Dramatic pause

CRITICAL: You MUST respond with ONLY valid JSON matching this EXACT schema:
{
  "cuts": [
    {"start": 10.5, "end": 18.2, "reason": "Hook intro", "transition": "cut"},
    {"start": 45.0, "end": 57.1, "reason": "Key moment", "transition": "crossfade"}
  ],
  "order": [0, 1]
}

IMPORTANT: 
- Use "start" and "end" as field names, NOT "start_time" or "end_time"
- Keep 2-4 clips total (fewer = more coherent)
- VERIFY your total duration is under 30 seconds before responding!
"""
    
    # Build user message, including critic feedback if available
    user_parts = [f"Here is the video transcript:\n{segment_details}"]
    
    if critic_feedback:
        user_parts.append(f"\n\n⚠️ IMPORTANT - YOUR PREVIOUS SELECTION WAS REJECTED:\n{critic_feedback}")
        user_parts.append("\nPlease select DIFFERENT clips that address the feedback above and form a more coherent narrative.")
    
    user_message = "\n".join(user_parts)
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
    
    logger.log_event("llm_call", {
        "node": "analyze_transcript", 
        "attempt": attempt,
        "has_feedback": critic_feedback is not None
    })

    try:
        # Use robust structured output
        result: AnalysisResult = call_llm_with_structure(llm, messages, AnalysisResult)
        
        logger.log_event("llm_response", {"node": "analyze_transcript", "structured_output": result.dict()})
        
        # Convert Pydantic model to dict for state
        cuts_data = [cut.dict() for cut in result.cuts]
        
        # Reorder if 'order' is provided
        if result.order:
            ordered_cuts = [cuts_data[i] for i in result.order if i < len(cuts_data)]
        else:
            ordered_cuts = cuts_data
        
        # --- Enforce 30-second maximum duration ---
        MAX_DURATION = 30.0
        total_duration = sum(cut['end'] - cut['start'] for cut in ordered_cuts)
        
        if total_duration > MAX_DURATION:
            print(f"   ⚠️  Total duration ({total_duration:.1f}s) exceeds {MAX_DURATION}s limit")
            print(f"   🔧 Trimming clips to fit within {MAX_DURATION}s...")
            
            # Strategy: Keep clips but trim them proportionally
            scale_factor = MAX_DURATION / total_duration
            trimmed_cuts = []
            running_duration = 0
            
            for cut in ordered_cuts:
                clip_duration = cut['end'] - cut['start']
                new_duration = clip_duration * scale_factor
                
                # Don't make clips shorter than 3 seconds (too short to be useful)
                if new_duration < 3.0 and running_duration + 3.0 <= MAX_DURATION:
                    new_duration = min(3.0, clip_duration)
                
                if running_duration + new_duration <= MAX_DURATION:
                    trimmed_cut = cut.copy()
                    trimmed_cut['end'] = cut['start'] + new_duration
                    trimmed_cuts.append(trimmed_cut)
                    running_duration += new_duration
                else:
                    # Can we fit a shortened version?
                    remaining = MAX_DURATION - running_duration
                    if remaining >= 3.0:
                        trimmed_cut = cut.copy()
                        trimmed_cut['end'] = cut['start'] + remaining
                        trimmed_cuts.append(trimmed_cut)
                    break
            
            ordered_cuts = trimmed_cuts
            total_duration = sum(cut['end'] - cut['start'] for cut in ordered_cuts)
            print(f"   ✅ Trimmed to {total_duration:.1f}s ({len(ordered_cuts)} clips)")
        
        # --- Print selected clips with transcript excerpts ---
        print(f"\n📋 SELECTED CLIPS ({len(ordered_cuts)} total):")
        print("-" * 50)
        total_duration = 0
        for i, cut in enumerate(ordered_cuts):
            start, end = cut['start'], cut['end']
            duration = end - start
            total_duration += duration
            
            # Find transcript text for this clip
            clip_text = _get_transcript_for_timerange(segments, start, end)
            clip_preview = clip_text[:80] + "..." if len(clip_text) > 80 else clip_text
            
            print(f"  Clip {i+1}: [{start:.1f}s - {end:.1f}s] ({duration:.1f}s)")
            print(f"    Reason: {cut.get('reason', 'N/A')}")
            print(f"    Transition: {cut.get('transition', 'cut')}")
            print(f"    Content: \"{clip_preview}\"")
            print()
        
        print(f"📊 Total output duration: {total_duration:.1f}s (max: {MAX_DURATION}s)")
        print("-" * 50)
            
        logger.log_event("node_end", {"node": "analyze_transcript", "output": {"cuts": ordered_cuts, "total_duration": total_duration}})
        return ordered_cuts
        
    except Exception as e:
        print(f"Error analyzing transcript (attempt {attempt}): {e}")
        logger.log_event("node_error", {"node": "analyze_transcript", "error": str(e), "attempt": attempt})
        # Fallback: Just take the first 30 seconds if Analysis fails
        return [{"start": 0, "end": 30, "reason": "Fallback - LLM Failed", "transition": "cut"}]


def analyze_with_critic(state: VideoState):
    """
    Analyzes transcript with critic feedback loop.
    
    This is the main analysis node that:
    1. Generates initial clip selection
    2. Has critic evaluate coherence
    3. Retries with feedback if rejected (up to 3 attempts)
    4. Returns best selection
    """
    transcript_text = state["transcript_text"]
    segments = state["transcript_segments"]
    
    print(f"\n{'='*60}")
    print(f"🎬 CONTENT ANALYSIS WITH CRITIC EVALUATION")
    print(f"{'='*60}")
    print(f"   Critic enabled: {ENABLE_CRITIC}")
    print(f"   Approval threshold: {CRITIC_APPROVAL_THRESHOLD}/10")
    print(f"   Max retry attempts: {CRITIC_MAX_RETRIES}")
    
    if not ENABLE_CRITIC:
        # Critic disabled - just do single analysis
        print("   ⚠️  Critic disabled - using single-pass analysis")
        cuts = analyze_transcript_with_feedback(transcript_text, segments)
        return {"cuts": cuts}
    
    # Initialize critic agent
    critic = create_critic_agent(
        approval_threshold=CRITIC_APPROVAL_THRESHOLD,
        verbose=True
    )
    
    best_cuts = None
    best_score = 0
    critic_feedback = None
    
    for attempt in range(1, CRITIC_MAX_RETRIES + 1):
        # Analyze transcript (with feedback if retry)
        cuts = analyze_transcript_with_feedback(
            transcript_text,
            segments,
            critic_feedback=critic_feedback,
            attempt=attempt
        )
        
        # We need a preliminary heading for critic context
        # Generate a quick heading for critic evaluation
        preliminary_heading = _generate_quick_heading(transcript_text)
        
        print(f"\n{'='*60}")
        print(f"🔍 CRITIC EVALUATION (Attempt {attempt}/{CRITIC_MAX_RETRIES})")
        print(f"{'='*60}")
        print(f"   Evaluating {len(cuts)} clips for narrative coherence...")
        
        # Have critic evaluate
        result, is_approved = critic.critique(
            cuts=cuts,
            transcript_segments=segments,
            transcript_text=transcript_text,
            heading=preliminary_heading,
            previous_feedback=critic_feedback,
            attempt_number=attempt,
            logger=logger
        )
        
        # Print detailed critic results
        print(f"\n📊 CRITIC VERDICT:")
        print(f"   Score: {result.coherence_score}/10 (threshold: {CRITIC_APPROVAL_THRESHOLD})")
        print(f"   Status: {'✅ APPROVED' if is_approved else '❌ REJECTED'}")
        print(f"   Narrative Flow: {result.narrative_flow}")
        print(f"   Context Alignment: {result.context_alignment}")
        if result.issues:
            print(f"   Issues Found:")
            for issue in result.issues:
                print(f"      • {issue}")
        if not is_approved and result.suggestions:
            print(f"   Suggestions: {result.suggestions}")
        
        # Track best attempt
        if result.coherence_score > best_score:
            best_score = result.coherence_score
            best_cuts = cuts
        
        if is_approved:
            print(f"\n✅ SELECTION APPROVED on attempt {attempt}")
            print(f"{'='*60}\n")
            return {"cuts": cuts}
        
        # Prepare feedback for retry
        critic_feedback = critic.format_feedback_for_retry(result)
        if attempt < CRITIC_MAX_RETRIES:
            print(f"\n🔄 Retrying with critic feedback...")
    
    # Max retries reached - use best attempt
    print(f"\n⚠️  MAX RETRIES REACHED")
    print(f"   Using best attempt with score: {best_score}/10")
    print(f"   Clips selected: {len(best_cuts) if best_cuts else 0}")
    print(f"{'='*60}\n")
    
    logger.log_event("critic_max_retries", {
        "best_score": best_score,
        "cuts_count": len(best_cuts) if best_cuts else 0
    })
    
    return {"cuts": best_cuts or cuts}


def _generate_quick_heading(transcript_text: str) -> str:
    """Generate a quick heading for critic context without full LLM call."""
    # Simple heuristic: use first 50 chars as context
    preview = transcript_text[:100].strip()
    if len(preview) > 50:
        preview = preview[:50] + "..."
    return f"Video about: {preview}"


def analyze_transcript(state: VideoState):
    """Legacy wrapper - now uses analyze_with_critic."""
    return analyze_with_critic(state)

def generate_heading(state: VideoState):
    """Generates a viral, witty heading for the video using LLM."""
    print(f"\n{'='*60}")
    print(f"📰 GENERATING HEADING")
    print(f"{'='*60}")
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
        print(f"\n   ✅ Generated: \"{heading}\"")
        print(f"{'='*60}\n")
        
        logger.log_event("node_end", {"node": "generate_heading", "output": {"heading": heading}})
        return {"heading": heading}
        
    except Exception as e:
        print(f"   ❌ Error generating heading: {e}")
        print(f"   Using fallback heading")
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


def _crop_to_vertical(clip, start_time=None):
    """Smart center-crop a clip to 9:16 aspect ratio using face tracking (if available)."""
    src_w, src_h = clip.w, clip.h
    src_aspect = src_w / src_h

    if src_aspect > TARGET_ASPECT:
        # Source is wider than 9:16 — crop width
        new_w = int(src_h * TARGET_ASPECT)
        x_offset = (src_w - new_w) // 2

        # Intelligent face-tracking crop if mediapipe is available
        try:
            import cv2
            import mediapipe as mp
            import numpy as np

            # Sample a frame from the middle of the clip to find the subject
            sample_time = clip.duration / 2
            frame = clip.get_frame(sample_time)
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            mp_face = mp.solutions.face_detection
            with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
                results = face_detection.process(frame_bgr)
                if results.detections:
                    # Find the largest/most prominent face
                    best_detection = max(results.detections, key=lambda d: d.location_data.relative_bounding_box.width)
                    bbox = best_detection.location_data.relative_bounding_box
                    
                    # Calculate center pixel of the face
                    face_center_x = int((bbox.xmin + bbox.width / 2) * src_w)
                    
                    # Align crop window with the face center, clamped to video bounds
                    x_offset = max(0, min(src_w - new_w, face_center_x - (new_w // 2)))
                    print(f"[Smart Crop] Found face at x={face_center_x}, adjusting crop offset to {x_offset}")

        except ImportError:
            print("[Smart Crop] OpenCV or MediaPipe not installed. Falling back to center crop.")
            pass
        except Exception as e:
            print(f"[Smart Crop] OpenCV Error: {e}")
            pass

        clip = clip.cropped(x1=x_offset, y1=0, x2=x_offset + new_w, y2=src_h)
    elif src_aspect < TARGET_ASPECT:
        # Source is taller than 9:16 — crop height
        new_h = int(src_w / TARGET_ASPECT)
        y_offset = (src_h - new_h) // 2
        clip = clip.cropped(x1=0, y1=y_offset, x2=src_w, y2=y_offset + new_h)

    # Resize to exact target resolution
    clip = clip.resized((TARGET_WIDTH, TARGET_HEIGHT))
    return clip


def edit_video(state: VideoState):
    """Cuts and stitches the video with 9:16 vertical format, dynamic captions, transitions, and context heading."""
    print(f"\n{'='*60}")
    print(f"🎬 EDITING VIDEO")
    print(f"{'='*60}")
    print(f"   Clips to process: {len(state['cuts'])}")
    print(f"   Heading: \"{state.get('heading', 'N/A')}\"")
    print(f"   Caption style: {CAPTION_STYLE.value}")
    print(f"   Background music: {'Enabled' if ENABLE_BACKGROUND_MUSIC else 'Disabled'}")
    
    logger.log_event("node_start", {"node": "edit_video", "input": {"cuts_count": len(state["cuts"]), "heading": state.get("heading")}})
    
    video_path = state["input_video_path"]
    cuts = state["cuts"]
    output_path = "output.mp4"
    
    # MoviePy imports
    from moviepy.video import fx as vfx
    from moviepy import TextClip, CompositeVideoClip, VideoFileClip
    import numpy as np
    
    # Initialize caption styler
    caption_styler = CaptionStyler(
        style=CAPTION_STYLE,
        target_width=TARGET_WIDTH,
        target_height=TARGET_HEIGHT,
        enable_emojis=ENABLE_EMOJIS
    )

    try:
        original_clip = VideoFileClip(video_path)
        clips = []
        
        font_path = _resolve_font()

        # --- Font sizes proportional to target output width ---
        heading_font_size = max(18, int(TARGET_WIDTH * 0.04))     # ~43px on 1080w
        heading_stroke = max(1, int(TARGET_WIDTH * 0.0025))       # ~3px

        # 1. First Pass: Create all subclips and overlay styled subtitles
        print(f"\n📎 PROCESSING CLIPS:")
        print("-" * 50)
        
        for idx, cut in enumerate(cuts):
            start = cut["start"]
            end = cut["end"]
            start = max(0, start)
            end = min(original_clip.duration, end)
            duration = end - start
            
            print(f"   Clip {idx+1}/{len(cuts)}: [{start:.1f}s - {end:.1f}s] ({duration:.1f}s)")
            print(f"      Reason: {cut.get('reason', 'N/A')}")
            
            if end > start:
                clip = original_clip.subclipped(start, end)
                
                # Crop each clip to 9:16 vertical, pass start time for debugging if needed
                clip = _crop_to_vertical(clip, start)
                
                # --- Dynamic Styled Caption Overlay using CaptionStyler ---
                subtitle_clips = create_styled_captions(
                    transcript_segments=state["transcript_segments"],
                    clip_start=start,
                    clip_end=end,
                    style=CAPTION_STYLE,
                    font_path=font_path,
                    target_width=TARGET_WIDTH,
                    target_height=TARGET_HEIGHT,
                    enable_emojis=ENABLE_EMOJIS
                )
                
                if subtitle_clips:
                    clip = CompositeVideoClip([clip] + subtitle_clips, size=(TARGET_WIDTH, TARGET_HEIGHT))
                    print(f"      Captions: {len(subtitle_clips)} text overlays added")
                
                clips.append(clip)
        
        print("-" * 50)
        
        if clips:
            final_clips_with_effects = []
            
            # 2. Apply Transitions
            for i in range(len(clips)):
                current_clip = clips[i]
                transition_type = cuts[i].get("transition", "cut") if i < len(cuts) else "cut"
                
                if i < len(clips) - 1:
                    next_clip = clips[i+1]
                    
                    if transition_type == "crossfade":
                        min_dur = min(current_clip.duration, next_clip.duration)
                        duration = min(1.0, min_dur / 2.0)
                        current_clip = current_clip.with_effects([vfx.FadeOut(duration)])
                        clips[i+1] = next_clip.with_effects([vfx.FadeIn(duration)])
                    elif transition_type == "fade_to_black":
                        duration = 0.5
                        current_clip = current_clip.with_effects([vfx.FadeOut(duration)])
                        clips[i+1] = next_clip.with_effects([vfx.FadeIn(duration)])
                
                final_clips_with_effects.append(current_clip)
            
            print(f"Concatenating {len(final_clips_with_effects)} clips")
            final_clip = concatenate_videoclips(final_clips_with_effects, method="compose")
            
            # 3. Semi-transparent top bar for heading (8% of height)
            bar_height = int(TARGET_HEIGHT * 0.08)  # ~154px on 1920h
            
            heading = state.get("heading")
            if heading:
                try:
                    # Create a semi-transparent dark gradient bar
                    def make_gradient_frame(t):
                        """Creates a top-to-bottom dark gradient bar with alpha."""
                        frame = np.zeros((bar_height, TARGET_WIDTH, 3), dtype=np.uint8)
                        for row in range(bar_height):
                            # Gradient from opacity ~0.85 at top to ~0.3 at bottom
                            alpha = 0.85 - (0.55 * row / bar_height)
                            frame[row, :] = int(alpha * 255 * 0.15)  # Dark tint
                        return frame
                    
                    gradient_bar = ColorClip(size=(TARGET_WIDTH, bar_height), color=(0, 0, 0))
                    gradient_bar = gradient_bar.with_opacity(0.65).with_duration(final_clip.duration)
                    gradient_bar = gradient_bar.with_position((0, 0))
                    
                    heading_clip = TextClip(
                        text=heading, 
                        font_size=heading_font_size, 
                        color='white', 
                        font=font_path,
                        stroke_color='black', 
                        stroke_width=heading_stroke,
                        method='caption', 
                        size=(int(TARGET_WIDTH * 0.9), bar_height),
                        text_align='center' 
                    )
                    heading_clip = heading_clip.with_position(('center', 0)).with_duration(final_clip.duration)
                    
                    final_clip = CompositeVideoClip(
                        [final_clip, gradient_bar, heading_clip],
                        size=(TARGET_WIDTH, TARGET_HEIGHT)
                    )
                    print(f"Added heading overlay: {heading}")
                except Exception as e:
                    print(f"Could not add heading overlay: {e}")
                    logger.log_event("warning", {"node": "edit_video", "warning": f"Heading overlay failed: {e}", "heading": heading})

            # 4. Add background music with audio ducking (if enabled and available)
            if ENABLE_BACKGROUND_MUSIC and MUSIC_AGENT_AVAILABLE:
                try:
                    print(f"\n🎵 ADDING BACKGROUND MUSIC")
                    print("-" * 50)
                    llm = ModelFactory.get_model(
                        provider=os.getenv("LLM_PROVIDER", "openrouter"),
                        model_name=os.getenv("LLM_MODEL", "meta-llama/llama-3.2-3b-instruct:free"),
                        temperature=0.7
                    )
                    final_clip = add_background_music(
                        video_clip=final_clip,
                        transcript_segments=state["transcript_segments"],
                        transcript_text=state["transcript_text"],
                        llm=llm,
                        logger=logger
                    )
                    print("-" * 50)
                except Exception as e:
                    print(f"   ⚠️  Could not add background music: {e}")
                    logger.log_event("warning", {"node": "edit_video", "warning": f"Background music failed: {e}"})

            print(f"\n📹 ENCODING FINAL VIDEO ({EXPORT_MODE.upper()} MODE)")
            print("-" * 50)
            settings = ENCODING_PRESETS[EXPORT_MODE]
            print(f"   Output: {output_path}")
            print(f"   Mode: {EXPORT_MODE}")
            print(f"   Resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}")
            print(f"   Duration: {final_clip.duration:.1f}s")
            print(f"   FPS: {settings['fps']}")
            print(f"   Preset: {settings['preset']}")
            print(f"   CRF: {settings['crf']} (higher = faster/smaller)")
            print(f"   Threads: {settings['threads']}")
            if EXPORT_MODE == "production" and USE_HARDWARE_ENCODING:
                print(f"   Hardware: VAAPI enabled (will try Intel GPU)")
            print("-" * 50)
            
            # Use optimized encoding with VAAPI fallback
            _encode_video_fast(final_clip, output_path, mode=EXPORT_MODE)
            
            print(f"\n{'='*60}")
            print(f"✅ VIDEO COMPLETE!")
            print(f"{'='*60}")
            print(f"   Output file: {output_path}")
            print(f"   Duration: {final_clip.duration:.1f}s")
            print(f"   Mode: {EXPORT_MODE}")
            if EXPORT_MODE == "preview":
                print(f"   💡 Tip: Set EXPORT_MODE='production' for final quality")
            print(f"{'='*60}\n")
            
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
workflow.add_node("analyze", analyze_transcript)
workflow.add_node("generate_heading", generate_heading)
workflow.add_node("edit_video", edit_video)

workflow.set_entry_point("extract_audio")

workflow.add_edge("extract_audio", "transcribe")
workflow.add_edge("transcribe", "analyze")
workflow.add_edge("analyze", "generate_heading")
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
    parser.add_argument("--preview", action="store_true", help="Preview mode: fast encoding at lower quality (480p)")
    parser.add_argument("--production", action="store_true", help="Production mode: high quality encoding (720p)")
    parser.add_argument("--no-critic", action="store_true", help="Disable critic agent for faster processing")
    parser.add_argument("--no-music", action="store_true", help="Disable background music")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file '{args.video_path}' not found.")
        sys.exit(1)

    # Apply command-line overrides using globals
    if args.preview:
        globals()['EXPORT_MODE'] = "preview"
        globals()['TARGET_WIDTH'] = PREVIEW_WIDTH
        globals()['TARGET_HEIGHT'] = PREVIEW_HEIGHT
    elif args.production:
        globals()['EXPORT_MODE'] = "production"
        globals()['TARGET_WIDTH'] = PRODUCTION_WIDTH
        globals()['TARGET_HEIGHT'] = PRODUCTION_HEIGHT
    
    if args.no_critic:
        globals()['ENABLE_CRITIC'] = False
    
    if args.no_music:
        globals()['ENABLE_BACKGROUND_MUSIC'] = False

    # Print startup banner
    print(f"\n{'='*60}")
    print(f"🎬 PRISM - AI Video Shorts Generator")
    print(f"{'='*60}")
    print(f"   Input: {args.video_path}")
    print(f"   Export Mode: {EXPORT_MODE.upper()}")
    print(f"   Resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}")
    print(f"   Critic: {'Enabled' if ENABLE_CRITIC else 'Disabled'}")
    print(f"   Music: {'Enabled' if ENABLE_BACKGROUND_MUSIC else 'Disabled'}")
    if args.dev:
        print(f"   Dev Mode: Transcript caching enabled")
    print(f"{'='*60}\n")
    
    initial_state = {"input_video_path": args.video_path, "dev_mode": args.dev}
    
    try:
        final_state = app.invoke(initial_state)
        print(f"Video processing complete! Output saved to: {final_state['output_video_path']}")
        logger.log_event("run_complete", {"output_video_path": final_state['output_video_path'], "dev_mode": args.dev, "export_mode": EXPORT_MODE})
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        logger.log_event("run_failed", {"error": str(e)})
    finally:
        # Cleanup temp files
        if os.path.exists("temp_audio.mp3"):
            os.remove("temp_audio.mp3")
            print("Cleaned up temp_audio.mp3")
