import json
import time
from typing import Type, TypeVar, List, Optional, Any, Dict, Literal, Union
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

# Try to import tenacity for retries, otherwise use a simple loop
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    HAS_TENACITY = True
except ImportError:
    HAS_TENACITY = False

T = TypeVar("T", bound=BaseModel)

# --- Pydantic Models for Structured Output ---

class VideoCut(BaseModel):
    start: float = Field(description="Start time in seconds. Use field name 'start', not 'start_time'.")
    end: float = Field(description="End time in seconds. Use field name 'end', not 'end_time'.")
    reason: str = Field(description="Brief reason for selecting this clip")
    transition: Literal["cut", "crossfade", "fade_to_black"] = Field(
        description="Transition type to next clip", 
        default="cut"
    )
    is_hook: bool = Field(
        description="Whether this is the hook/opening segment (first 3 seconds are critical)",
        default=False
    )
    energy_level: Literal["low", "medium", "high", "climax"] = Field(
        description="Energy/intensity level of this segment for zoom/speed effects",
        default="medium"
    )

    @model_validator(mode='before')
    @classmethod
    def normalize_field_names(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Handle start_time -> start
            if 'start_time' in data and 'start' not in data:
                data['start'] = data.pop('start_time')
            # Handle end_time -> end
            if 'end_time' in data and 'end' not in data:
                data['end'] = data.pop('end_time')
            # Handle content/text -> reason
            if 'content' in data and 'reason' not in data:
                data['reason'] = data.pop('content')
            if 'text' in data and 'reason' not in data:
                data['reason'] = data.pop('text')
        return data


class CaptionWord(BaseModel):
    """A word with styling information for animated captions."""
    word: str = Field(description="The word text")
    is_emphasis: bool = Field(description="Whether this word should be emphasized (larger, different color)", default=False)
    emoji: Optional[str] = Field(description="Optional emoji to show with this word", default=None)


class CaptionGroup(BaseModel):
    """A group of 2-4 words to display together as animated caption."""
    words: List[CaptionWord] = Field(description="Words in this caption group")
    start_time: float = Field(description="Start time in seconds")
    end_time: float = Field(description="End time in seconds")


class CaptionAnalysis(BaseModel):
    """LLM-analyzed captions with emphasis and emoji suggestions."""
    groups: List[CaptionGroup] = Field(description="Caption groups for the video")


class AnalysisResult(BaseModel):
    cuts: List[VideoCut] = Field(description="List of video cuts with start, end, reason, transition")
    order: Optional[List[int]] = Field(description="Optional order of cuts by index", default=None)
    hook_segment_index: Optional[int] = Field(
        description="Index of the segment that should be used as the hook (first 3 seconds are critical for retention)",
        default=0
    )

class HeadingResult(BaseModel):
    heading: str = Field(description="A short viral heading, 5-7 words max")


class CriticResult(BaseModel):
    """Result from the Critic Agent evaluating clip selection coherence."""
    approved: bool = Field(description="Whether the clip selection is approved")
    coherence_score: int = Field(ge=1, le=10, description="Coherence score from 1-10")
    narrative_flow: str = Field(description="Assessment of how well clips flow together")
    context_alignment: str = Field(description="How well clips align with the heading/context")
    issues: List[str] = Field(default_factory=list, description="List of specific issues found")
    suggestions: str = Field(description="Specific suggestions for improvement if rejected")
    
    @model_validator(mode='before')
    @classmethod
    def normalize_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Handle alternative field names
            if 'narrative_issues' in data and 'issues' not in data:
                data['issues'] = data.pop('narrative_issues')
            if 'feedback' in data and 'suggestions' not in data:
                data['suggestions'] = data.pop('feedback')
            if 'score' in data and 'coherence_score' not in data:
                data['coherence_score'] = data.pop('score')
        return data

# --- Helper Functions ---

def strip_markdown_json(text: str) -> str:
    """Strip markdown code blocks from JSON response."""
    import re
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ```
    pattern = r'^```(?:json)?\s*\n?(.*?)\n?```$'
    match = re.match(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text

def parse_json_response(text: str, schema: Type[T]) -> T:
    """Parse JSON response and validate against schema."""
    cleaned = strip_markdown_json(text)
    data = json.loads(cleaned)
    return schema.model_validate(data)

# --- Robust LLM Caller ---

def call_llm_with_structure(
    llm: ChatOpenAI, 
    messages: List[BaseMessage], 
    schema: Type[T],
    max_retries: int = 3
) -> T:
    """
    Calls the LLM and parses JSON response into the Pydantic schema.
    Uses manual JSON parsing with markdown stripping for reliability.
    """
    # Use manual parsing directly - more reliable across different LLM providers
    # Many LLMs (including Gemini) wrap JSON in markdown code blocks
    return _call_with_manual_parsing(llm, messages, schema, max_retries)

def _call_with_manual_parsing(llm: ChatOpenAI, messages: List[BaseMessage], schema: Type[T], max_retries: int) -> T:
    """Fallback: call LLM and manually parse JSON response."""
    last_exception: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages)
            # Handle different response content types
            if hasattr(response, 'content'):
                content = response.content
                # If content is a list (multimodal response), extract text parts
                if isinstance(content, list):
                    text_parts = [part if isinstance(part, str) else part.get("text", "") for part in content]
                    content = "".join(text_parts)
                else:
                    content = str(content)
            else:
                content = str(response)
            return parse_json_response(content, schema)
        except Exception as e:
            print(f"Manual parsing attempt {attempt + 1} failed: {e}")
            last_exception = e
            time.sleep(2 ** attempt)
    if last_exception is not None:
        raise last_exception
    raise RuntimeError("Max retries reached with no exception captured")

def _call_with_tenacity(structured_llm: Runnable, messages: List[BaseMessage], max_retries: int):
    """Retries using Tenacity library."""
    if not HAS_TENACITY:
        raise ImportError("tenacity is not installed")
    
    @retry(  # type: ignore[name-defined]
        stop=stop_after_attempt(max_retries),  # type: ignore[name-defined]
        wait=wait_exponential(multiplier=1, min=2, max=10),  # type: ignore[name-defined]
        retry=retry_if_exception_type((ValidationError, ValueError, Exception)),  # type: ignore[name-defined]
        reraise=True
    )
    def invoke():
        return structured_llm.invoke(messages)
        
    return invoke()

def _call_with_simple_retry(structured_llm: Runnable, messages: List[BaseMessage], max_retries: int):
    """Simple retry loop if tenacity is not available."""
    last_exception: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return structured_llm.invoke(messages)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            last_exception = e
            time.sleep(2 ** attempt)
    
    if last_exception is not None:
        raise last_exception
    raise RuntimeError("Max retries reached with no exception captured")


