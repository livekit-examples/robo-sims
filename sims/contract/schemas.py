"""JSON schemas registered on LiveKit publications (not in discovery)."""

DURATION = {"type": "object", "required": ["sec", "nanosec"], "additionalProperties": False,
            "properties": {"sec": {"type": "integer", "minimum": 0},
                           "nanosec": {"type": "integer", "minimum": 0, "maximum": 999999999}}}
NUMBERS = {"type": "array", "items": {"type": "number"}}
NAMES = {"type": "array", "items": {"type": "string"}, "uniqueItems": True}
HEADER = {"type": "object", "additionalProperties": False,
          "properties": {"stamp": DURATION, "frame_id": {"type": "string"}}}
JOINT_STATE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
    "required": ["header", "name", "position", "velocity", "effort"],
    "additionalProperties": False,
    "properties": {"header": HEADER, "name": NAMES, "position": NUMBERS,
                   "velocity": NUMBERS, "effort": NUMBERS},
}
TRAJECTORY_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
    "required": ["joint_names", "points"], "additionalProperties": False,
    "properties": {"header": HEADER, "joint_names": NAMES, "points": {
        "type": "array", "maxItems": 256, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["positions", "time_from_start"],
            "properties": {"positions": NUMBERS, "time_from_start": DURATION,
                           **{key: {"type": "array", "maxItems": 0}
                              for key in ("velocities", "accelerations", "effort")}},
        }}},
}
