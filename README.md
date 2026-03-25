# PRISM - AI-Powered Viral Video Creator

> Transform long-form videos into viral short-form content automatically using multi-agent AI

PRISM uses cutting-edge AI to analyze your videos, identify the most engaging moments, and produce polished TikTok/Instagram Reels-ready content with professional captions, smart cropping, and background music.

## Features

- **AI Content Analysis**: LLM-powered viral moment detection
- **Smart Cropping**: Face-tracking 9:16 vertical formatting
- **Dynamic Captions**: 7 style presets (Hormozi, TikTok, News, etc.)
- **Background Music**: AI mood matching with professional audio ducking
- **Auto Headings**: Contextual headlines for viewer engagement
- **Multi-Provider LLM**: OpenRouter, OpenAI, Gemini support

## Quick Start

### 1. Install Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Or using uv (recommended)
uv sync
```

### 2. Set Up Environment

Create a `.env` file with your API key:

```bash
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Optional: Use different LLM providers
# LLM_PROVIDER=openai
# LLM_MODEL=gpt-4
# OPENAI_API_KEY=sk-...
```

### 3. Run PRISM

```bash
# Basic usage
python video_graph.py path/to/your/video.mp4

# Development mode (caches transcripts for faster iteration)
python video_graph.py path/to/your/video.mp4 --dev
```

### 4. Output

Your viral short will be saved as `output.mp4` in the current directory.

## Configuration

Edit the constants at the top of `video_graph.py`:

```python
# Caption style: TIKTOK, HORMOZI, NEWS, MINIMAL, KARAOKE, IMPACT, GRADIENT
CAPTION_STYLE = CaptionStyle.HORMOZI

# Enable/disable features
ENABLE_BACKGROUND_MUSIC = True
ENABLE_EMOJIS = False
```

## Adding Background Music

1. Run the music agent setup:
```bash
python music_agent.py
```

2. Add royalty-free music files to the created folders:
```
music_library/
├── energetic/    # High-energy tracks
├── dramatic/     # Cinematic, serious
├── uplifting/    # Feel-good, positive
├── chill/        # Lo-fi, relaxed
├── intense/      # Tension, confrontation
├── inspirational/# Motivational
└── neutral/      # Generic background
```

Recommended sources for royalty-free music:
- [Pixabay Music](https://pixabay.com/music)
- [YouTube Audio Library](https://studio.youtube.com/channel/audio)
- [Mixkit](https://mixkit.co/free-stock-music/)

## Caption Styles

| Style | Description | Best For |
|-------|-------------|----------|
| `HORMOZI` | Big word pops, yellow text | Motivational, business |
| `TIKTOK` | Classic centered bold | General viral content |
| `NEWS` | Lower-third banner | Informational, serious |
| `MINIMAL` | Clean, subtle | Professional, corporate |
| `KARAOKE` | Word-by-word highlight | Music, lyrics |
| `IMPACT` | High contrast, dramatic | Attention-grabbing |
| `GRADIENT` | Colorful gradient text | Trendy, Gen-Z |

## Debugging

### View Execution Flow

```bash
streamlit run visualize.py
```

### Check Logs

Execution logs are saved to `execution_logs.jsonl`.

### Cached Transcripts

Transcripts are cached in `.dev_cache/` by video filename for faster re-runs.

## Project Structure

```
PRISM/
├── video_graph.py      # Main pipeline
├── llm_core.py         # LLM structured outputs
├── model_factory.py    # Multi-provider LLM
├── music_agent.py      # Background music AI
├── caption_styler.py   # Dynamic captions
├── visualize.py        # Streamlit debugger
├── requirements.txt    # Dependencies
└── music_library/      # Your music files
```

## Requirements

- Python 3.11+
- FFmpeg (for video processing)
- GPU recommended (for Whisper transcription)

## Troubleshooting

**"No font found" warning**: Install system fonts or the script will use defaults.

**Slow transcription**: Ensure CUDA is available for GPU acceleration, or use `--dev` mode to cache transcripts.

**LLM errors / Empty responses**: The free OpenRouter model may be rate-limited or unavailable. Try these fixes:

1. **Switch to a different free model** in your `.env`:
   ```bash
   LLM_MODEL=meta-llama/llama-3.2-3b-instruct:free
   ```

2. **Recommended free models** (in order of reliability):
   - `meta-llama/llama-3.2-3b-instruct:free`
   - `google/gemma-2-9b-it:free`
   - `mistralai/mistral-7b-instruct:free`
   - `qwen/qwen-2-7b-instruct:free`

3. **Use OpenAI directly** (requires paid API key):
   ```bash
   LLM_PROVIDER=openai
   LLM_MODEL=gpt-4o-mini
   OPENAI_API_KEY=sk-...
   ```

**JSON parsing errors**: Usually means the LLM returned malformed output. The system will retry automatically. If persistent, try a different model.

---

Built with LangGraph + OpenAI Whisper + MoviePy | Multi-Agent AI Architecture