"""Fixed calibration contract checks without MuJoCo or LiveKit."""

import copy
import unittest

from sims.contract.camera_info import validate_camera_info
from sims.contract.config import CONFIGS, Configuration


def camera_info():
    return {"front": {"track_name": "front-stereo", "width": 1280, "height": 480, "views": {
        eye: {"frame_id": f"front_{eye}_optical",
            "image_region": {"x": index * 640, "y": 0, "width": 640, "height": 480},
            "intrinsics": {"fx": 500, "fy": 500, "cx": 320, "cy": 240,
                "distortion_model": "none", "distortion_coefficients": []},
            "extrinsics": {"parent_frame_id": "front_mount", "translation": [index * 0.06, 0, 0],
                "rotation_xyzw": [0, 0, 0, 1]}}
        for index, eye in enumerate(("left", "right"))}}}


class CameraInfoTests(unittest.TestCase):
    def test_optional_mono_stereo_and_shared_physical_camera_frames(self):
        validate_camera_info({})
        info = camera_info()
        validate_camera_info(info)
        info["resized"] = copy.deepcopy(info["front"])
        info["resized"]["track_name"] = "second-video-track"
        validate_camera_info(info)
        mono = info["front"]
        mono["views"] = {"mono": mono["views"]["left"]}
        validate_camera_info(info)

    def test_distortion_coefficients_match_the_model(self):
        for model, count in (("none", 0), ("plumb_bob", 5), ("rational_polynomial", 8), ("equidistant", 4)):
            with self.subTest(model=model):
                info = camera_info()
                intrinsic = info["front"]["views"]["left"]["intrinsics"]
                intrinsic.update(distortion_model=model, distortion_coefficients=[0] * count)
                validate_camera_info(info)
                intrinsic["distortion_coefficients"].append(0)
                with self.assertRaises(ValueError):
                    validate_camera_info(info)

    def test_rejects_invalid_calibration_in_discovery(self):
        original = Configuration.load(CONFIGS / "trossen-wxai.bindings.json", 12)
        for problem in ("width", "views", "region", "overlap", "frame", "focal", "principal",
                        "distortion", "translation", "quaternion", "parent", "track"):
            config = copy.deepcopy(original)
            info = config.discovery["camera_info"] = camera_info()
            camera = info["front"]
            left = camera["views"]["left"]
            if problem == "width":
                camera["width"] = True
            elif problem == "views":
                del camera["views"]["right"]
            elif problem == "region":
                left["image_region"]["width"] = 2000
            elif problem == "overlap":
                camera["views"]["right"]["image_region"]["x"] = 0
            elif problem == "frame":
                camera["views"]["right"]["frame_id"] = left["frame_id"]
            elif problem == "focal":
                left["intrinsics"]["fx"] = 0
            elif problem == "principal":
                left["intrinsics"]["cx"] = float("nan")
            elif problem == "distortion":
                left["intrinsics"]["distortion_coefficients"] = [1]
            elif problem == "translation":
                left["extrinsics"]["translation"] = [0, 0]
            elif problem == "quaternion":
                left["extrinsics"]["rotation_xyzw"] = [0, 0, 0, 0]
            elif problem == "parent":
                left["extrinsics"]["parent_frame_id"] = left["frame_id"]
            else:
                info["duplicate"] = copy.deepcopy(camera)
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                config.validate()

    def test_known_field_cannot_be_null_or_a_list(self):
        for invalid in (None, [], {"bad-id": {}}, {"front": None}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_camera_info(invalid)
