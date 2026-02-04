import os
import json
import warnings
from typing import TypedDict, List, Annotated
import operator

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from moviepy import VideoFileClip, concatenate_videoclips
import whisper
import torch

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Check for API Key
if not os.getenv("OPENROUTER_API_KEY"):
    raise ValueError("OPENROUTER_API_KEY not found in .env file")

# --- State Definition ---
class VideoState(TypedDict):
    input_video_path: str
    audio_path: str
    transcript_text: str
    transcript_segments: List[dict] # List of {start: float, end: float, text: str}
    cuts: List[dict] # {start: float, end: float, reason: str}
    output_video_path: str

# --- Nodes ---

def extract_audio(state: VideoState):
    """Extracts audio from the input video."""
    print("--- Extracting Audio ---")
    video_path = state["input_video_path"]
    audio_path = "temp_audio.mp3"
    
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, logger=None)
        video.close()
        return {"audio_path": audio_path}
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return {"audio_path": None} 

def transcribe_audio(state: VideoState):
    """Transcribes audio using local Whisper model."""
    print("--- Transcribing Audio ---")
    audio_path = state["audio_path"]
    
    if not audio_path or not os.path.exists(audio_path):
        raise FileNotFoundError("Audio file not found or extraction failed.")

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        model = whisper.load_model("base", device=device)
        result = model.transcribe(audio_path)
        return {
            "transcript_text": result["text"],
            "transcript_segments": result["segments"]
        }
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        raise e

def analyze_transcript(state: VideoState):
    """Analyzes transcript using OpenRouter to find viral cuts."""
    print("--- Analyzing Transcript ---")
    transcript_text = state["transcript_text"]
    segments = state["transcript_segments"]
    
    # Initialize OpenRouter Chat Model
    llm = ChatOpenAI(
        model="z-ai/glm-4.5-air:free",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.7
    )
    
    # Detailed Context for the LLM
    segment_details = "\n".join([f"[{s['start']:.2f}-{s['end']:.2f}]: {s['text']}" for s in segments])
    
    system_prompt = """
    You are an expert video editor creating viral shorts for Gen Z.
    Your goal is to select the most engaging, funny, or insightful segments from the transcript to create a fast-paced 30-60 second video.
    
    Output ONLY valid JSON in the following format:
    {
        "cuts": [
            {"start": 10.5, "end": 15.2, "reason": "Funny intro"},
            {"start": 45.0, "end": 50.1, "reason": "Key punchline"}
        ],
        "order": [0, 1] 
    }
    The "order" field specifies the index of the cuts in the final sequence. You can reorder them if it makes the narrative better.
    Ensure the start and end times match the provided transcript segments accurately.
    Merge adjacent segments if they flow together.
    """
    
    user_message = f"Here is the transcript with timestamps:\n{segment_details}"
    
    try:
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_message)])
        content = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        # print the result
        print(result)

        # Sort/Reorder cuts based on 'order'
        ordered_cuts = [result["cuts"][i] for i in result.get("order", range(len(result["cuts"])))]
        
        return {"cuts": ordered_cuts}
    except Exception as e:
        print(f"Error analyzing transcript: {e}")
        # Fallback: Just take the first 30 seconds if Analysis fails
        return {"cuts": [{"start": 0, "end": 30, "reason": "Fallback"}]}

def edit_video(state: VideoState):
    """Cuts and stitches the video based on analysis."""
    print("--- Editing Video ---")
    video_path = state["input_video_path"]
    cuts = state["cuts"]
    output_path = "output.mp4"
    
    try:
        original_clip = VideoFileClip(video_path)
        clips = []
        
        for cut in cuts:
            start = cut["start"]
            end = cut["end"]
            # Ensure boundaries are within video duration
            start = max(0, start)
            end = min(original_clip.duration, end)
            
            if end > start:
                clip = original_clip.subclipped(start, end)
                clips.append(clip)
        
        if clips:
            final_clip = concatenate_videoclips(clips)
            final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
            final_clip.close()
        original_clip.close()
        
        return {"output_video_path": output_path}
    except Exception as e:
        print(f"Error editing video: {e}")
        raise e

# --- Graph Construction ---

workflow = StateGraph(VideoState)

workflow.add_node("extract_audio", extract_audio)
workflow.add_node("transcribe", transcribe_audio)
workflow.add_node("analyze", analyze_transcript)
workflow.add_node("edit_video", edit_video)

workflow.set_entry_point("extract_audio")

workflow.add_edge("extract_audio", "transcribe")
workflow.add_edge("transcribe", "analyze")
workflow.add_edge("analyze", "edit_video")
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
    except Exception as e:
        print(f"An error occurred during execution: {e}")
