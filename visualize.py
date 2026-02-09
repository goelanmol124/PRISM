import streamlit as st
import json
import pandas as pd
from datetime import datetime

st.set_page_config(layout="wide", page_title="Video Graph Flow Visualizer")

LOG_FILE = "execution_logs.jsonl"

def load_logs():
    logs = []
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                logs.append(json.loads(line))
    except FileNotFoundError:
        st.warning(f"Log file '{LOG_FILE}' not found. Run video_graph.py first.")
    return logs

logs = load_logs()

if not logs:
    st.stop()

# Group logs by run_id
runs = {}
for log in logs:
    run_id = log["run_id"]
    if run_id not in runs:
        runs[run_id] = []
    runs[run_id].append(log)

# Sidebar: Select Run
st.sidebar.title("Execution History")
run_options = []
for run_id, run_logs in runs.items():
    start_time = run_logs[0]["timestamp"]
    # Parse timestamp for better display
    dt = datetime.fromisoformat(start_time)
    formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
    run_options.append(f"{formatted_time} ({run_id[:8]})")

selected_run_label = st.sidebar.selectbox("Select a Run", run_options, index=len(run_options)-1)
selected_run_index = run_options.index(selected_run_label)
selected_run_id = list(runs.keys())[selected_run_index]
current_run_logs = runs[selected_run_id]

# Main View
st.title(f"Run Details: {selected_run_label}")

# Timeline
st.markdown("### Execution Flow")

for log in current_run_logs:
    event_type = log["event_type"]
    data = log["data"]
    timestamp = log["timestamp"]
    
    if event_type == "node_start":
        node_name = data.get("node")
        with st.expander(f"🔵 **Node Started**: {node_name}", expanded=True):
            st.caption(f"Time: {timestamp}")
            st.json(data.get("input"))
            
    elif event_type == "llm_call":
        node_name = data.get("node")
        with st.expander(f"🧠 **LLM Call**: {node_name}", expanded=False):
            st.caption(f"Time: {timestamp}")
            st.markdown("**System Prompt:**")
            st.code(data.get("system_prompt"), language="markdown")
            st.markdown("**User Message Preview:**")
            st.code(data.get("user_message_preview"), language="markdown")

    elif event_type == "llm_response":
        node_name = data.get("node")
        with st.expander(f"💡 **LLM Response**: {node_name}", expanded=False):
            st.caption(f"Time: {timestamp}")
            st.code(data.get("raw_response"), language="json")

    elif event_type == "node_end":
        node_name = data.get("node")
        with st.expander(f"✅ **Node Finished**: {node_name}", expanded=True):
            st.caption(f"Time: {timestamp}")
            st.json(data.get("output"))

    elif event_type == "node_error":
        node_name = data.get("node")
        st.error(f"❌ **Error in {node_name}**: {data.get('error')}")

    elif event_type == "run_complete":
        st.success(f"🎉 **Run Complete**! Output: {data.get('output_video_path')}")

    elif event_type == "run_failed":
        st.error(f"💀 **Run Failed**: {data.get('error')}")
