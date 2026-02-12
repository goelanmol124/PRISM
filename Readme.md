# PRISM: AI-Powered Automated Video Editor

PRISM is an intelligent video processing pipeline that automatically edits long-form videos into engaging short-form content. It uses LLMs to analyze transcripts, identify viral moments, and "smart cut" the video, all orchestrated via a graph-based workflow.

## 📂 Project Structure

Here's an overview of the key files in the codebase and how they connect:

### Core Logic
*   **`video_graph.py`**: The **main orchestration script**. It defines the processing pipeline using `langgraph`.
    *   **Flow**: Extract Audio -> Transcribe (Whisper) -> Analyze (LLM) -> Edit (MoviePy).
    *   **Key Class**: `VideoState` defines the data passing through the graph.
*   **`llm_core.py`**: Handles **LLM interactions and structured output**.
    *   Defines Pydantic models (`VideoCut`, `AnalysisResult`) for strict JSON validation.
    *   Includes retry logic (`tenacity` or simple loops) to ensure robust API calls.
*   **`model_factory.py`**: A **factory pattern** for creating LLM instances.
    *   Supports `OpenAI`, `OpenRouter`, and `Gemini` providers seamlessly.

### Utilities
*   **`visualize.py`**: A **Streamlit app** to visualize the execution flow.
    *   Reads `execution_logs.jsonl` and shows a timeline of LLM calls, prompts, and responses.
*   **`requirements.txt`**: List of Python dependencies.
*   **`.dev_cache/`**: Directory where transcripts are cached to speed up development iterations.

---

## 🚀 How to Run

### 1. Setup Environment
Ensure you have Python 3.10+ installed.

```bash
# Install dependencies
pip install -r requirements.txt

# Create a .env file with your API keys
# Required: OPENROUTER_API_KEY (or OPENAI_API_KEY / GOOGLE_API_KEY)
touch .env
```

### 2. Run the Editor
To process a video, run the main script:

```bash
python video_graph.py path/to/your/video.mp4
```
*The script will output `output.mp4` in the same directory.*

### 3. Visualize Execution
To see exactly what the AI did (prompts, decisions, errors):

```bash
streamlit run visualize.py
```

---

## 🔮 Roadmap / Next Steps

We are currently at **Phase 1** (Audio-based Cutting). The next immediate goals (Phase 2) are:

### Implementation Plan

#### 1. Visual Context (Screenshots)
To make smarter cuts, the AI needs to "see" the video, not just "hear" it.
*   **Goal**: Extract frames at regular intervals (e.g., every 5s) and pass them to a Vision LLM (like Gemini 1.5 Flash or GPT-4o).
*   **Action**: 
    1.  Update `extract_audio` or add a new node `extract_frames`.
    2.  Update `analyze_transcript` to accept image inputs.
    3.  Feed visual descriptions (e.g., "Speaker is holding a product", "Screen shows code") into the prompt.

#### 2. Background Music
To increase engagement, soft background music should be added.
*   **Goal**: Mix a royalty-free track with the final video, using "audio ducking" (lowering music volume when speech is present).
*   **Action**:
    1.  Add a `add_background_music` node after `edit_video`.
    2.  Use `moviepy` to overlay audio tracks.
    3.  Implement a simple volume envelope (ducking) based on the audio waveform.

### Future Ideas
*   **Auto-Captions**: Burn stylized captions into the video (like TikTok/Reels).
*   **Face Tracking**: Automatically crop specifically for 9:16 vertical video if the input is horizontal.
