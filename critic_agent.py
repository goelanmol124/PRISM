"""
PRISM Critic Agent - Narrative Coherence Evaluator
===================================================
Evaluates whether selected video clips form a coherent narrative
and provides feedback for iterative improvement.

The Critic Agent ensures that:
1. Selected clips tell a complete, coherent story
2. The narrative flows logically from clip to clip
3. The heading/context accurately represents the content
4. The overall message is clear and engaging
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from llm_core import call_llm_with_structure, CriticResult
from model_factory import ModelFactory


@dataclass
class ClipInfo:
    """Information about a selected clip for critique."""
    index: int
    start: float
    end: float
    reason: str
    transcript_excerpt: str


class CriticAgent:
    """
    AI-powered Critic Agent that evaluates narrative coherence.
    
    The critic reviews:
    1. Do the clips form a complete story arc?
    2. Is there logical flow between clips?
    3. Does the heading accurately represent the content?
    4. Are there gaps or redundancies in the narrative?
    """
    
    # Minimum coherence score to approve (1-10 scale)
    APPROVAL_THRESHOLD = 7
    
    def __init__(
        self,
        approval_threshold: int = 7,
        verbose: bool = True
    ):
        self.approval_threshold = approval_threshold
        self.verbose = verbose
    
    def _get_clip_transcripts(
        self,
        cuts: List[Dict[str, Any]],
        transcript_segments: List[Dict[str, Any]]
    ) -> List[ClipInfo]:
        """
        Extract transcript excerpts for each selected clip.
        
        Args:
            cuts: List of selected cuts with start/end times
            transcript_segments: Full transcript with timing
            
        Returns:
            List of ClipInfo with transcript excerpts
        """
        clip_infos = []
        
        for idx, cut in enumerate(cuts):
            start = cut.get("start", 0)
            end = cut.get("end", 0)
            reason = cut.get("reason", "No reason provided")
            
            # Find transcript segments that overlap with this clip
            excerpt_parts = []
            for seg in transcript_segments:
                seg_start = seg.get("start", 0)
                seg_end = seg.get("end", 0)
                
                # Check for overlap
                if seg_start < end and seg_end > start:
                    excerpt_parts.append(seg.get("text", ""))
            
            excerpt = " ".join(excerpt_parts).strip()
            if len(excerpt) > 300:
                excerpt = excerpt[:300] + "..."
            
            clip_infos.append(ClipInfo(
                index=idx + 1,
                start=start,
                end=end,
                reason=reason,
                transcript_excerpt=excerpt or "[No transcript for this segment]"
            ))
        
        return clip_infos
    
    def _format_clips_for_prompt(self, clip_infos: List[ClipInfo]) -> str:
        """Format clip information for the LLM prompt."""
        lines = []
        for clip in clip_infos:
            lines.append(f"""
Clip {clip.index} [{clip.start:.1f}s - {clip.end:.1f}s]:
  Selection Reason: {clip.reason}
  Content: "{clip.transcript_excerpt}"
""")
        return "\n".join(lines)
    
    def critique(
        self,
        cuts: List[Dict[str, Any]],
        transcript_segments: List[Dict[str, Any]],
        transcript_text: str,
        heading: Optional[str] = None,
        previous_feedback: Optional[str] = None,
        attempt_number: int = 1,
        logger=None
    ) -> Tuple[CriticResult, bool]:
        """
        Evaluate the coherence of selected clips.
        
        Args:
            cuts: List of selected cuts with start/end times and reasons
            transcript_segments: Full transcript with word-level timing
            transcript_text: Full transcript text
            heading: Optional heading/context for the video
            previous_feedback: Feedback from previous critique (if retry)
            attempt_number: Current attempt number (1-3)
            logger: Optional structured logger
            
        Returns:
            Tuple of (CriticResult, is_approved)
        """
        if self.verbose:
            print(f"--- Critic Agent (Attempt {attempt_number}/3) ---")
        
        if logger:
            logger.log_event("node_start", {
                "node": "critic_agent",
                "attempt": attempt_number,
                "cuts_count": len(cuts),
                "heading": heading
            })
        
        # Get LLM
        llm = ModelFactory.get_model(
            provider=os.getenv("LLM_PROVIDER", "openrouter"),
            model_name=os.getenv("LLM_MODEL", "meta-llama/llama-3.2-3b-instruct:free"),
            temperature=0.4  # Lower temperature for more consistent evaluation
        )
        
        # Prepare clip information
        clip_infos = self._get_clip_transcripts(cuts, transcript_segments)
        clips_formatted = self._format_clips_for_prompt(clip_infos)
        
        # Build system prompt
        system_prompt = """You are a professional video editor and storytelling expert.
Your job is to evaluate whether a set of video clips will create a coherent, engaging short-form video.

HARD REQUIREMENT: Total duration MUST be 30 seconds or less. If total exceeds 30s, auto-reject.

EVALUATION CRITERIA:
1. **Duration Check**: Total duration must be ≤30 seconds (REJECT if over)
2. **Narrative Completeness**: Do the clips tell a complete story with a clear beginning, middle, and end?
3. **Logical Flow**: Do the clips connect logically? Is there a clear progression of ideas?
4. **Context Alignment**: Does the content match the heading/topic? Does it convey the background context?
5. **Engagement**: Will viewers understand what's happening and stay engaged?
6. **No Gaps**: Are there missing pieces that would confuse viewers?
7. **No Redundancy**: Are clips meaningfully different from each other?

SCORING GUIDELINES:
- 9-10: Excellent - Clips form a perfect, compelling narrative under 30s
- 7-8: Good - Minor improvements possible but overall coherent
- 5-6: Fair - Some coherence issues that should be addressed
- 3-4: Poor - Significant gaps or confusion in the narrative
- 1-2: Very Poor - Clips don't form any coherent story OR exceed 30s

APPROVAL THRESHOLD: Score of 7 or higher = APPROVED

CRITICAL: Respond with ONLY valid JSON in this EXACT format:
{
    "approved": true,
    "coherence_score": 8,
    "narrative_flow": "The clips progress logically from introduction to key testimony to conclusion",
    "context_alignment": "The content matches the heading about disability testimony",
    "issues": ["Minor: Could use a stronger opening hook"],
    "suggestions": "Consider starting with the most impactful statement to hook viewers immediately"
}

If rejecting (score < 7), be SPECIFIC about what clips to change and why.
"""
        
        # Build user message
        user_parts = []
        
        if heading:
            user_parts.append(f"VIDEO HEADING/CONTEXT: \"{heading}\"")
        
        user_parts.append(f"\nFULL TRANSCRIPT SUMMARY (first 1000 chars):\n{transcript_text[:1000]}...")
        user_parts.append(f"\nSELECTED CLIPS ({len(cuts)} total):")
        user_parts.append(clips_formatted)
        
        if previous_feedback:
            user_parts.append(f"\n⚠️ PREVIOUS REJECTION FEEDBACK:\n{previous_feedback}")
            user_parts.append("\nPlease evaluate if the new selection addresses these concerns.")
        
        user_parts.append("\nEvaluate these clips for narrative coherence. Approve if score >= 7.")
        
        user_message = "\n".join(user_parts)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        
        if logger:
            logger.log_event("llm_call", {
                "agent": "CriticAgent",
                "attempt": attempt_number,
                "clips_count": len(cuts),
                "has_previous_feedback": previous_feedback is not None
            })
        
        try:
            result = call_llm_with_structure(llm, messages, CriticResult)
            
            # Determine approval based on score threshold
            is_approved = result.coherence_score >= self.approval_threshold and result.approved
            
            if self.verbose:
                status = "✅ APPROVED" if is_approved else "❌ REJECTED"
                print(f"[Critic] {status} (Score: {result.coherence_score}/10)")
                print(f"[Critic] Narrative Flow: {result.narrative_flow}")
                if result.issues:
                    print(f"[Critic] Issues: {', '.join(result.issues)}")
                if not is_approved:
                    print(f"[Critic] Suggestions: {result.suggestions}")
            
            if logger:
                logger.log_event("llm_response", {
                    "agent": "CriticAgent",
                    "result": result.dict(),
                    "is_approved": is_approved
                })
                logger.log_event("node_end", {
                    "node": "critic_agent",
                    "approved": is_approved,
                    "score": result.coherence_score
                })
            
            return result, is_approved
            
        except Exception as e:
            print(f"[Critic] Error during evaluation: {e}")
            
            if logger:
                logger.log_event("node_error", {
                    "node": "critic_agent",
                    "error": str(e)
                })
            
            # On error, create a default "needs improvement" result
            return CriticResult(
                approved=False,
                coherence_score=5,
                narrative_flow="Could not evaluate - using default assessment",
                context_alignment="Unknown",
                issues=["Evaluation failed - consider re-selecting clips"],
                suggestions="Try selecting clips that clearly relate to each other and tell a complete story"
            ), False
    
    def format_feedback_for_retry(self, result: CriticResult) -> str:
        """
        Format critic feedback for the content analyzer retry.
        
        Args:
            result: CriticResult from critique
            
        Returns:
            Formatted feedback string for the analyzer
        """
        feedback_parts = [
            f"CRITIC FEEDBACK (Score: {result.coherence_score}/10):",
            f"",
            f"Narrative Assessment: {result.narrative_flow}",
            f"Context Alignment: {result.context_alignment}",
        ]
        
        if result.issues:
            feedback_parts.append(f"")
            feedback_parts.append(f"Issues Found:")
            for issue in result.issues:
                feedback_parts.append(f"  - {issue}")
        
        if result.suggestions:
            feedback_parts.append(f"")
            feedback_parts.append(f"Required Improvements: {result.suggestions}")
        
        return "\n".join(feedback_parts)


def create_critic_agent(
    approval_threshold: int = 7,
    verbose: bool = True
) -> CriticAgent:
    """
    Factory function to create a CriticAgent instance.
    
    Args:
        approval_threshold: Minimum coherence score to approve (1-10)
        verbose: Whether to print status messages
        
    Returns:
        Configured CriticAgent instance
    """
    return CriticAgent(
        approval_threshold=approval_threshold,
        verbose=verbose
    )


if __name__ == "__main__":
    # Demo usage
    print("Critic Agent - Narrative Coherence Evaluator")
    print("=" * 50)
    print("\nThis agent evaluates whether selected video clips form a coherent narrative.")
    print("\nUsage in pipeline:")
    print("  1. Content Analyzer selects clips")
    print("  2. Critic Agent evaluates coherence")
    print("  3. If rejected, Analyzer retries with feedback")
    print("  4. Max 3 attempts before accepting best effort")
