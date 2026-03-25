# AGENTS.md - AI Agent Guidelines for PRISM

## Project Overview

PRISM is an AI-powered video editing pipeline that transforms long-form videos into short-form content for TikTok, Instagram Reels, and YouTube Shorts. It uses LLMs to analyze transcripts, identify viral moments, and perform "smart cuts" via a LangGraph-based workflow.

**Tech Stack**: Python 3.11+, LangChain, LangGraph, MoviePy, Whisper, Pydantic, OpenRouter/OpenAI

## Build & Run Commands

### Package Management (uses `uv`, lockfile: `uv.lock`)
```bash
uv sync                              # Install dependencies with uv (preferred)
pip install -r requirements.txt      # Or use pip
```

### Running the Application
```bash
python video_graph.py path/to/video.mp4        # Process a video
python video_graph.py path/to/video.mp4 --dev  # Dev mode (caches transcripts)
streamlit run visualize.py                     # Run execution visualizer
```

### Testing
```bash
pytest                                           # Run all tests
pytest tests/test_llm_core.py                    # Single test file
pytest tests/test_llm_core.py::test_function -v  # Single test function
pytest --cov=. --cov-report=html                 # With coverage
```

### Linting & Formatting
```bash
mypy .           # Type checking
ruff check .     # Linting
ruff format .    # Formatting
```

## File Organization

| File | Purpose |
|------|---------|
| `video_graph.py` | Main orchestration (LangGraph workflow, video effects, editing) |
| `llm_core.py` | LLM interactions, Pydantic models, structured output parsing |
| `model_factory.py` | Factory pattern for LLM providers (OpenAI, OpenRouter, Gemini) |
| `visualize.py` | Streamlit debugging UI for execution logs |
| `.dev_cache/` | Cached transcripts and keyframes (gitignored) |
| `assets/music/` | Background music tracks (.mp3, .wav, .m4a) |

## Code Style Guidelines

### Imports (order: stdlib -> third-party -> local)
```python
import os
import json
from typing import TypedDict, List, Any, Dict, Optional, Type, TypeVar

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field, model_validator

from llm_core import call_llm_with_structure, AnalysisResult
from model_factory import ModelFactory
```

### Type Hints
- **Always use type hints** for function signatures
- Use `TypedDict` for LangGraph state objects
- Use Pydantic `BaseModel` for LLM structured outputs

```python
def call_llm_with_structure(
    llm: ChatOpenAI, messages: List[BaseMessage], schema: Type[T], max_retries: int = 3
) -> T:

class VideoState(TypedDict):
    input_video_path: str
    transcript_text: str
    cuts: List[dict]
```

### Naming Conventions
- **Functions**: `snake_case` - `extract_audio`, `analyze_transcript`
- **Classes**: `PascalCase` - `VideoState`, `ModelFactory`
- **Constants**: `UPPER_SNAKE_CASE` - `TARGET_WIDTH`, `CACHE_DIR`
- **Private helpers**: `_prefix` - `_resolve_font`, `_crop_to_vertical`

### Pydantic Models for LLM Output
```python
class VideoCut(BaseModel):
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    reason: str = Field(description="Brief reason for selecting this clip")
    transition: Literal["cut", "crossfade", "fade_to_black"] = Field(default="cut")

    @model_validator(mode='before')
    @classmethod
    def normalize_field_names(cls, data: Any) -> Any:
        """Handle LLM response variations (e.g., start_time -> start)"""
        if isinstance(data, dict):
            if 'start_time' in data and 'start' not in data:
                data['start'] = data.pop('start_time')
        return data
```

### Error Handling
- Implement retry logic for LLM calls (exponential backoff)
- Always provide fallback behavior when LLM calls fail
- Log errors with structured logging via `logger.log_event()`

```python
try:
    result = call_llm_with_structure(llm, messages, AnalysisResult)
except Exception as e:
    logger.log_event("node_error", {"node": "analyze_transcript", "error": str(e)})
    return {"cuts": [{"start": 0, "end": 30, "reason": "Fallback - LLM Failed"}]}
```

### LangGraph Node Pattern
```python
def node_name(state: VideoState) -> dict:
    """Docstring describing the node's purpose."""
    print("--- Node Name ---")
    logger.log_event("node_start", {"node": "node_name", "input": {...}})
    
    try:
        result = {"key": value}
        logger.log_event("node_end", {"node": "node_name", "output": result})
        return result
    except Exception as e:
        logger.log_event("node_error", {"node": "node_name", "error": str(e)})
        raise e  # Or return fallback
```

## Environment Variables

Store in `.env` (gitignored):

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | Yes* | - | Required for OpenRouter provider |
| `OPENAI_API_KEY` | Yes* | - | Required for OpenAI provider |
| `GOOGLE_API_KEY` | Yes* | - | Required for Gemini provider |
| `LLM_PROVIDER` | No | `openrouter` | LLM provider to use |
| `LLM_MODEL` | No | `z-ai/glm-4.5-air:free` | Model name |
| `VISION_PROVIDER` | No | `openrouter` | Vision LLM provider |
| `VISION_MODEL` | No | `openai/gpt-4o-mini` | Vision model name |
| `ENABLE_VISION_ANALYSIS` | No | `true` | Enable/disable vision analysis |

*At least one API key required depending on provider used.

## Video Processing Constants
```python
TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920
TARGET_ASPECT = TARGET_WIDTH / TARGET_HEIGHT  # 9:16 for vertical video
```

## Common Tasks

### Adding a New LLM Provider
1. Add method `_get_<provider>_model()` to `ModelFactory` in `model_factory.py`
2. Handle API key retrieval from environment
3. Return a LangChain-compatible chat model

### Adding a New Graph Node
1. Define the node function in `video_graph.py` following the node pattern above
2. Add node: `workflow.add_node("node_name", node_function)`
3. Add edges: `workflow.add_edge("previous_node", "node_name")`
4. Update `VideoState` TypedDict if new state fields are needed

### Adding a New Pydantic Schema for LLM
1. Define the model in `llm_core.py` with Field descriptions
2. Add `@model_validator` if field name normalization is needed
3. Use with `call_llm_with_structure(llm, messages, YourSchema)`
