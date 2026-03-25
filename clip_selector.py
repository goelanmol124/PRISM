"""
Clip Selection Module for PRISM
===============================
LLM-powered analysis for identifying viral moments and generating contextual headings.

Features:
- Hook optimization (attention-grabbing openings)
- Energy level detection for dynamic effects
- Viral cut selection with timing optimization
- Contextual heading generation
"""

import os
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage

from model_factory import ModelFactory
from llm_core import call_llm_with_structure, AnalysisResult, HeadingResult


def analyze_transcript_for_cuts(
    transcript_text: str,
    transcript_segments: List[Dict[str, Any]],
    logger = None
) -> List[Dict[str, Any]]:
    """
    Analyze transcript using LLM to find viral cuts with hook optimization and energy levels.
    
    Returns:
        List of cut dictionaries with start, end, reason, energy_level, is_hook, transition
    """
    print("--- Analyzing Transcript (Enhanced) ---")
    if logger:
        logger.log_event("node_start", {"node": "analyze_transcript", "input": {"transcript_preview": transcript_text[:200]}})
    
    # Initialize LLM via ModelFactory
    llm = ModelFactory.get_model(
        provider=os.getenv("LLM_PROVIDER", "openrouter"),
        model_name=os.getenv("LLM_MODEL", "z-ai/glm-4.5-air:free"),
        temperature=0.7
    )
    
    # Detailed Context for the LLM
    segment_details = "\n".join([f"[{s['start']:.2f}-{s['end']:.2f}]: {s['text']}" for s in transcript_segments])
    
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
    
    if logger:
        logger.log_event("llm_call", {
            "node": "analyze_transcript", 
            "system_prompt": system_prompt, 
            "user_message_preview": user_message[:500] + "..."
        })

    try:
        # Use robust structured output
        result: AnalysisResult = call_llm_with_structure(llm, messages, AnalysisResult)
        
        if logger:
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
        
        if logger:
            logger.log_event("node_end", {"node": "analyze_transcript", "output": {"cuts": ordered_cuts, "hook": hook_cut}})
        
        return ordered_cuts
        
    except Exception as e:
        print(f"Error analyzing transcript (retries failed): {e}")
        if logger:
            logger.log_event("node_error", {"node": "analyze_transcript", "error": str(e)})
        # Fallback: Just take the first 30 seconds if Analysis fails
        return [{"start": 0, "end": 30, "reason": "Fallback - LLM Failed", "transition": "cut", "is_hook": True, "energy_level": "medium"}]


def generate_contextual_heading(
    transcript_text: str,
    logger = None
) -> str:
    """
    Generate a viral, witty heading for the video using LLM.
    
    Returns:
        Contextual heading string
    """
    print("--- Generating Heading ---")
    if logger:
        logger.log_event("node_start", {"node": "generate_heading", "input": {"transcript_preview": transcript_text[:200]}})
    
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
    
    user_message = f"Here is the video transcript:\n{transcript_text[:2000]}..."  # Truncate for efficiency if needed
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
    
    if logger:
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
        
        if logger:
            logger.log_event("node_end", {"node": "generate_heading", "output": {"heading": heading}})
        
        return heading
        
    except Exception as e:
        print(f"Error generating heading (retries failed): {e}")
        if logger:
            logger.log_event("node_error", {"node": "generate_heading", "error": str(e)})
        return "Economics 101"  # Fallback