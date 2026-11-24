"""Validate fixed camera calibration in a discovery document; standard library only."""

import math
import re

DISTORTION_COUNTS = {"none": 0, "plumb_bob": 5, "rational_polynomial": 8, "equidistant": 4}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def vector(value, length):
    return isinstance(value, list) and len(value) == length and all(finite(v) for v in value)


def text(value):
    return isinstance(value, str) and bool(value.strip())


def integer(value, minimum):
    return finite(value) and value == int(value) and minimum <= value <= 2**53 - 1


def validate_camera_info(value):
    if not isinstance(value, dict):
        raise ValueError("camera_info must be an object")
    tracks = set()
    for key, camera in value.items():
        if not isinstance(key, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", key) \
                or not isinstance(camera, dict):
            raise ValueError("invalid camera_info entry")
        name = camera.get("track_name")
        if not text(name) or name in tracks:
            raise ValueError("camera_info needs unique track names")
        tracks.add(name)
        width, height = camera.get("width"), camera.get("height")
        if not integer(width, 1) or not integer(height, 1):
            raise ValueError("camera_info needs positive integer image dimensions")
        views = camera.get("views")
        if not isinstance(views, dict) or set(views) not in ({"mono"}, {"left", "right"}):
            raise ValueError("camera views must contain mono, or both left and right")
        frames = set()
        for view in views.values():
            if not isinstance(view, dict):
                raise ValueError("camera view must be an object")
            frame = view.get("frame_id")
            if not text(frame) or frame in frames:
                raise ValueError("camera_info needs unique optical frame IDs")
            frames.add(frame)
            region = view.get("image_region")
            if not isinstance(region, dict) or not all(integer(region.get(k), minimum)
                    for k, minimum in (("x", 0), ("y", 0), ("width", 1), ("height", 1))) \
                    or region["x"] + region["width"] > width or region["y"] + region["height"] > height:
                raise ValueError("camera image_region must fit inside the video image")
            intrinsics = view.get("intrinsics")
            if not isinstance(intrinsics, dict) or not all(finite(intrinsics.get(k))
                    for k in ("fx", "fy", "cx", "cy")) \
                    or intrinsics["fx"] <= 0 or intrinsics["fy"] <= 0:
                raise ValueError("invalid camera intrinsics")
            model = intrinsics.get("distortion_model")
            if not isinstance(model, str) or model not in DISTORTION_COUNTS \
                    or not vector(intrinsics.get("distortion_coefficients"), DISTORTION_COUNTS[model]):
                raise ValueError("invalid camera distortion model or coefficients")
            extrinsics = view.get("extrinsics")
            if not isinstance(extrinsics, dict) or not text(extrinsics.get("parent_frame_id")) \
                    or extrinsics["parent_frame_id"] == frame \
                    or not vector(extrinsics.get("translation"), 3) \
                    or not vector(extrinsics.get("rotation_xyzw"), 4) \
                    or abs(math.hypot(*extrinsics["rotation_xyzw"]) - 1) > 1e-4:
                raise ValueError("invalid camera extrinsics or unit quaternion")
        if "left" in views:
            a, b = (views[eye]["image_region"] for eye in ("left", "right"))
            if a["x"] < b["x"] + b["width"] and b["x"] < a["x"] + a["width"] \
                    and a["y"] < b["y"] + b["height"] and b["y"] < a["y"] + a["height"]:
                raise ValueError("stereo image regions cannot overlap")
