"""
Structured Logger Module for PRISM
==================================
JSON-based logging system for execution tracking and debugging.
"""

import json
import uuid
import datetime
from typing import Dict, Any


class StructuredLogger:
    """Structured logger that writes events to JSONL format."""
    
    def __init__(self, log_file: str = "execution_logs.jsonl"):
        self.log_file = log_file
        self.run_id = str(uuid.uuid4())
        self.log_event("run_start", {"timestamp": datetime.datetime.now().isoformat()})

    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an event with structured data."""
        entry = {
            "run_id": self.run_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "event_type": event_type,
            "data": data
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")