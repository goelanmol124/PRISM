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

class AnalysisResult(BaseModel):
    cuts: List[VideoCut] = Field(description="List of video cuts with start, end, reason, transition")
    order: Optional[List[int]] = Field(description="Optional order of cuts by index", default=None)

class HeadingResult(BaseModel):
    heading: str = Field(description="A short viral heading, 5-7 words max")

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
    last_exception = None
    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)
            return parse_json_response(content, schema)
        except Exception as e:
            print(f"Manual parsing attempt {attempt + 1} failed: {e}")
            last_exception = e
            time.sleep(2 ** attempt)
    raise last_exception

def _call_with_tenacity(structured_llm: Runnable, messages: List[BaseMessage], max_retries: int):
    """Retries using Tenacity library."""
    
    @retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ValidationError, ValueError, Exception)),
        reraise=True
    )
    def invoke():
        return structured_llm.invoke(messages)
        
    return invoke()

def _call_with_simple_retry(structured_llm: Runnable, messages: List[BaseMessage], max_retries: int):
    """Simple retry loop if tenacity is not available."""
    last_exception = None
    for attempt in range(max_retries):
        try:
            return structured_llm.invoke(messages)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            last_exception = e
            time.sleep(2 ** attempt)
    
    raise last_exception


