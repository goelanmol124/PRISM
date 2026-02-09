import json
import time
from typing import Type, TypeVar, List, Optional, Any, Dict, Literal
try:
    from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError
except ImportError:
    from pydantic import BaseModel, Field, ValidationError
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
    start: float = Field(description="Start time of the clip in seconds")
    end: float = Field(description="End time of the clip in seconds")
    reason: str = Field(description="Reason for selecting this clip")
    transition: Literal["cut", "crossfade", "fade_to_black"] = Field(
        description="Transition type to the next clip", 
        default="cut"
    )

class AnalysisResult(BaseModel):
    cuts: List[VideoCut] = Field(description="List of selected video cuts")
    order: Optional[List[int]] = Field(description="Order of cuts by index (optional)", default=None)

class HeadingResult(BaseModel):
    heading: str = Field(description="A viral, witty heading for the video")

# --- Robust LLM Caller ---

def call_llm_with_structure(
    llm: ChatOpenAI, 
    messages: List[BaseMessage], 
    schema: Type[T],
    max_retries: int = 3
) -> T:
    """
    Calls the LLM and enforces a structured output based on the Pydantic schema.
    Includes retry logic for validation errors or API issues.
    """
    
    # Check if with_structured_output is available (LangChain >= 0.1.0 with langchain-openai)
    if hasattr(llm, "with_structured_output"):
        structured_llm = llm.with_structured_output(schema)
    else:
        # Fallback to PydanticOutputParser if the method is missing (older versions)
        # However, for simplicity in this prototype, we'll assume updated libs or raise an error.
        # Ideally, we'd implement a PydanticOutputParser fallback here.
        raise ImportError("LangChain version is too old. Please upgrade langchain-openai to use with_structured_output.")

    if HAS_TENACITY:
        return _call_with_tenacity(structured_llm, messages, max_retries)
    else:
        return _call_with_simple_retry(structured_llm, messages, max_retries)

def _call_with_tenacity(structured_llm: Runnable, messages: List[BaseMessage], max_retries: int):
    """Retries using Tenacity library."""
    
    # We define the retry decorator dynamically or use a wrapper function
    @retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ValidationError, ValueError, Exception)), # Broad exception for prototype
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
            time.sleep(2 ** attempt) # Exponential backoff: 1s, 2s, 4s...
    
    raise last_exception

