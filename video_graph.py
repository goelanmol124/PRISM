import os
import json
import warnings
from typing import TypedDict, List, Any, Dict
import datetime
import uuid

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from moviepy import VideoFileClip, concatenate_videoclips
import whisper
import torch
from llm_core import call_llm_with_structure, AnalysisResult, HeadingResult
from model_factory import ModelFactory

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Check for API Key
if not os.getenv("OPENROUTER_API_KEY"):
    raise ValueError("OPENROUTER_API_KEY not found in .env file")

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

# --- Nodes ---

def extract_audio(state: VideoState):
    """Extracts audio from the input video."""
    print("--- Extracting Audio ---")
    logger.log_event("node_start", {"node": "extract_audio", "input": {"video_path": state["input_video_path"]}})
    
    video_path = state["input_video_path"]
    audio_path = "temp_audio.mp3"
    
    try:
        video = VideoFileClip(video_path)
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
    """Transcribes audio using local Whisper model."""
    print("--- Transcribing Audio ---")
    logger.log_event("node_start", {"node": "transcribe_audio", "input": {"audio_path": state["audio_path"]}})
    
    audio_path = state["audio_path"]
    
    if not audio_path or not os.path.exists(audio_path):
        error_msg = "Audio file not found or extraction failed."
        logger.log_event("node_error", {"node": "transcribe_audio", "error": error_msg})
        raise FileNotFoundError(error_msg)

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
        logger.log_event("node_end", {"node": "transcribe_audio", "output": {"transcript_text_preview": result["text"][:100], "segment_count": len(result["segments"])}})
        return output
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        logger.log_event("node_error", {"node": "transcribe_audio", "error": str(e)})
        raise e

def analyze_transcript(state: VideoState):
    """Analyzes transcript using OpenRouter/LLM to find viral cuts and transition types."""
    print("--- Analyzing Transcript ---")
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
    You are an expert video editor creating viral shorts for Gen Z.
    Your goal is to select the most engaging, funny, or insightful segments from the transcript to create a fast-paced 30-60 second video.
    
    For each cut, decide the best transition TO the NEXT clip:
    - "cut": Fast-paced, standard dialogue. Use this for 80% of transitions.
    - "crossfade": Smooth flow between related topics.
    - "fade_to_black": Dramatic pause or scene change.
    
    The "transition" field specifies how this clip should transition into the NEXT clip in the sequence.
    Merge adjacent segments if they flow together.
    """
    
    user_message = f"Here is the video transcript:\n{segment_details}"
    
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
        
        # Reorder if 'order' is provided
        if result.order:
            ordered_cuts = [cuts_data[i] for i in result.order if i < len(cuts_data)]
        else:
            ordered_cuts = cuts_data
            
        logger.log_event("node_end", {"node": "analyze_transcript", "output": {"cuts": ordered_cuts}})
        return {"cuts": ordered_cuts}
        
    except Exception as e:
        print(f"Error analyzing transcript (retries failed): {e}")
        logger.log_event("node_error", {"node": "analyze_transcript", "error": str(e)})
        # Fallback: Just take the first 30 seconds if Analysis fails
        return {"cuts": [{"start": 0, "end": 30, "reason": "Fallback - LLM Failed", "transition": "cut"}]}

def generate_heading(state: VideoState):
    """Generates a viral, witty heading for the video using LLM."""
    print("--- Generating Heading ---")
    logger.log_event("node_start", {"node": "generate_heading", "input": {"transcript_preview": state["transcript_text"][:200]}})
    
    transcript_text = state["transcript_text"]
    
    llm = ModelFactory.get_model(
        provider=os.getenv("LLM_PROVIDER", "openrouter"),
        model_name=os.getenv("LLM_MODEL", "z-ai/glm-4.5-air:free"),
        temperature=0.8
    )
    
    system_prompt = """
    You are a viral content strategist for Gen Z on Instagram Reels and TikTok, specifically for the Indian market.
    Your goal is to create a SINGLE, short, punchy, and witty heading for a video about Economics/Finance.
    
    The heading should be:
    - Catchy and hook the viewer instantly.
    - Relevant to the content but with a fun, modern twist.
    - Use Gen Z slang or internet culture references where appropriate (but keep it understandable).
    - MAX 5-7 words.
    - NO hashtags.
    - "POV:", "Me when:", or question formats work well.
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

def edit_video(state: VideoState):
    """Cuts and stitches the video based on analysis, with smart transitions."""
    print("--- Editing Video ---")
    logger.log_event("node_start", {"node": "edit_video", "input": {"cuts_count": len(state["cuts"]), "heading": state.get("heading")}})
    
    video_path = state["input_video_path"]
    cuts = state["cuts"]
    output_path = "output.mp4"
    
    # MoviePy v2.x imports for effects
    from moviepy.video import fx as vfx
    from moviepy import TextClip, CompositeVideoClip

    try:
        original_clip = VideoFileClip(video_path)
        clips = []
        
        # 1. First Pass: Create all subclips
        for cut in cuts:
            start = cut["start"]
            end = cut["end"]
            start = max(0, start)
            end = min(original_clip.duration, end)
            
            if end > start:
                clip = original_clip.subclipped(start, end)
                clips.append(clip)
        
        if clips:
            final_clips_with_effects = []
            
            # Iterate to apply transitions
            for i in range(len(clips)):
                current_clip = clips[i]
                
                # Default to 'cut' if processing last clip or error
                transition_type = cuts[i].get("transition", "cut") if i < len(cuts) else "cut"
                
                # Check if next clip exists
                if i < len(clips) - 1:
                    next_clip = clips[i+1]
                    
                    # Logic for transitions
                    if transition_type == "crossfade":
                        min_dur = min(current_clip.duration, next_clip.duration)
                        duration = min(1.0, min_dur / 2.0)
                        
                        # Apply fade effects for crossfade simulation
                        current_clip = current_clip.with_effects([vfx.FadeOut(duration)])
                        clips[i+1] = next_clip.with_effects([vfx.FadeIn(duration)])
                        
                    elif transition_type == "fade_to_black":
                        duration = 0.5
                        # Fade out current clip, fade in next clip
                        current_clip = current_clip.with_effects([vfx.FadeOut(duration)])
                        clips[i+1] = next_clip.with_effects([vfx.FadeIn(duration)])
                
                final_clips_with_effects.append(current_clip)
            
            print(f"Concatenating {len(final_clips_with_effects)} clips")

            final_clip = concatenate_videoclips(final_clips_with_effects, method="compose")
            
            # 4. Overlay Heading
            heading = state.get("heading")
            if heading:
                try:
                    # Create a TextClip - requires ImageMagick for some methods
                    txt_clip = TextClip(
                        text=heading, 
                        font_size=70, 
                        color='white', 
                        font='Arial-Bold',
                        stroke_color='black', 
                        stroke_width=2,
                        size=(int(final_clip.w * 0.8), None)
                    )
                    txt_clip = txt_clip.with_position(('center', 100)).with_duration(final_clip.duration)
                    
                    final_clip = CompositeVideoClip([final_clip, txt_clip])
                    print(f"Added heading overlay: {heading}")
                except Exception as e:
                    print(f"Could not add heading overlay (ImageMagick issue?): {e}")

            final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
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
    
    # Simple CLI
    if len(sys.argv) < 2:
        print("Usage: python video_graph.py <video_path>")
        sys.exit(1)
        
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        sys.exit(1)

    print(f"Processing video: {video_path}")
    initial_state = {"input_video_path": video_path}
    
    try:
        final_state = app.invoke(initial_state)
        print(f"Video processing complete! Output saved to: {final_state['output_video_path']}")
        logger.log_event("run_complete", {"output_video_path": final_state['output_video_path']})
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        logger.log_event("run_failed", {"error": str(e)})
    finally:
        # Cleanup temp files
        if os.path.exists("temp_audio.mp3"):
            os.remove("temp_audio.mp3")
            print("Cleaned up temp_audio.mp3")
