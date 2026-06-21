import re
from datetime import datetime
from typing import Dict, Optional


def parse_temporal(temporal_obj: Optional[Dict]) -> str:
    """Parse complex temporal elements with error handling."""
    if not temporal_obj:
        return "Timing not specified"
    try:
        if "age" in temporal_obj:
            return parse_iso_duration(temporal_obj["age"].get("iso8601duration"))
        if "timestamp" in temporal_obj:
            return datetime.fromisoformat(temporal_obj["timestamp"]).strftime(
                "%Y-%m-%d"
            )
        if "interval" in temporal_obj:
            start = temporal_obj["interval"].get("start", "unknown")
            end = temporal_obj["interval"].get("end", "unknown")
            return f"{start} to {end}"
        return "Timing information available"
    except Exception as e:
        return f"Timing information unavailable: {str(e)}"


def parse_iso_duration(duration: Optional[str]) -> str:
    """Convert ISO8601 duration to a human-readable format."""
    if not duration:
        return "Age unspecified"
    try:
        match = re.match(r"P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)D)?", duration)
        parts = []
        if match:
            if match.group(1):
                parts.append(f"{match.group(1)} years")
            if match.group(2):
                parts.append(f"{match.group(2)} months")
            if match.group(3):
                parts.append(f"{match.group(3)} days")
            return " ".join(parts) if parts else duration
        return duration
    except Exception as e:
        return f"Duration parsing failed: {str(e)}"
