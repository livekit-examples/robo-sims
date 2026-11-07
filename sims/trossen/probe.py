"""Manual LiveKit integration probe for the Trossen simulations.

    python -m sims.trossen.probe --robot <identity>

Run this against `python -m sims run trossen-workbench --protocol legacy`, with two operator tokens for
the same room. It exercises the operator protocol without starting robot code.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import secrets
import struct
from dataclasses import dataclass, field
from typing import Callable

from livekit import rtc
import numpy as np


SESSION = secrets.randbits(32)


def joint_command(target: str, sequence: int, value: float, joint_index: int = 0,
                  kind: int = 1) -> bytes:
    identity = target.encode("utf-8")
    return (struct.pack("<4sBBBBIIQd", b"TLJC", 1, kind, len(identity), joint_index,
                        SESSION, sequence, 0, value) + identity)


def motion_command(target: str, sequence: int, values: tuple[float, ...],
                   kind: bytes = b"TLTC", active: bool = True) -> bytes:
    identity = target.encode("utf-8")
    return (struct.pack("<4sBBBBIIQ6f", kind, 1, int(active), len(identity), 0,
                        SESSION, sequence, 0, *values) + identity)


@dataclass
class Observation:
    room: rtc.Room = field(default_factory=rtc.Room)
    telemetry: asyncio.Queue[tuple[str, tuple[float, ...]]] = field(default_factory=asyncio.Queue)
    videos: dict[str, tuple[int, int]] = field(default_factory=dict)
    stereo_difference: dict[str, float] = field(default_factory=dict)
    readers: list[asyncio.Task] = field(default_factory=list)
    closed: bool = False

    def watch(self) -> None:
        @self.room.on("data_track_published")
        def on_data(track: rtc.RemoteDataTrack) -> None:
            if not track.info.name.endswith(".joints.v1"):
                return
            stream = track.subscribe(buffer_size=32)

            async def read() -> None:
                try:
                    async for packet in stream:
                        payload = packet.payload
                        if len(payload) >= 16 + 7 * 24 and payload[:4] == b"TLJS":
                            arm = track.info.name.split(".")[3]
                            values = tuple(struct.unpack_from("<d", payload, 16 + 24 * i)[0]
                                           for i in range(7))
                            self.telemetry.put_nowait((arm, values))
                finally:
                    await stream.aclose()

            self.readers.append(asyncio.create_task(read()))

        @self.room.on("track_subscribed")
        def on_video(track: rtc.Track, publication: rtc.RemoteTrackPublication,
                     _participant: rtc.RemoteParticipant) -> None:
            if not isinstance(track, rtc.RemoteVideoTrack):
                return
            stream = rtc.VideoStream.from_track(
                track=track, capacity=2, format=rtc.VideoBufferType.RGB24)

            async def read() -> None:
                try:
                    async for packet in stream:
                        self.videos[publication.name] = (packet.frame.width, packet.frame.height)
                        if publication.name.endswith(".stereo"):
                            pixels = np.frombuffer(packet.frame.data, dtype=np.uint8).reshape(
                                packet.frame.height, packet.frame.width, 3)
                            midpoint = packet.frame.width // 2
                            self.stereo_difference[publication.name] = float(np.mean(np.abs(
                                pixels[:, :midpoint].astype(np.int16) -
                                pixels[:, midpoint:].astype(np.int16))))
                        break
                finally:
                    await stream.aclose()

            self.readers.append(asyncio.create_task(read()))

    async def wait_joint(self, arm: str, predicate: Callable[[tuple[float, ...]], bool],
                         timeout: float = 8) -> tuple[float, ...]:
        last = None
        async def next_match() -> tuple[float, ...]:
            nonlocal last
            while True:
                name, value = await self.telemetry.get()
                if name == arm:
                    last = value
                if name == arm and predicate(value):
                    return value
        try:
            return await asyncio.wait_for(next_match(), timeout)
        except TimeoutError as error:
            raise AssertionError(f"timed out waiting for {arm} joint state; last={last}") from error

    async def wait_videos(self, names: set[str], timeout: float = 12) -> None:
        async def all_received() -> None:
            while not names.issubset(self.videos):
                await asyncio.sleep(0.05)
        await asyncio.wait_for(all_received(), timeout)

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for task in self.readers:
            task.cancel()
        await asyncio.gather(*self.readers, return_exceptions=True)
        await self.room.disconnect()


async def rpc(room: rtc.Room, robot: str, method: str) -> str:
    return await room.local_participant.perform_rpc(
        destination_identity=robot, method=method, payload="")


async def probe(url: str, robot: str, controller_token: str, viewer_token: str,
                height: int, scene_width: int, scene_height: int) -> None:
    controller, viewer = Observation(), Observation()
    controller.watch()
    try:
        await controller.room.connect(url, controller_token)
        config = json.loads(await rpc(controller.room, robot, "get_robot_config"))
        assert config["schema_version"] == 1
        assert [arm["id"] for arm in config["arms"]] == ["left", "right"]
        assert not config["mobile_base"]
        names = {video["name"] for video in config["video_tracks"]}
        stereo = ("sim.left.stereo", "sim.center.stereo", "sim.right.stereo")
        assert names == {"sim.scene", *stereo}
        print("PASS: configuration advertises two arms and four cameras")

        await controller.wait_videos(names)
        assert controller.videos["sim.scene"] == (scene_width, scene_height)
        # Each ZED eye has the 16:10 sensor shape, so its width follows the height.
        eye_width = 2 * round(height * 1.6 / 2)
        for name in stereo:
            assert controller.videos[name] == (2 * eye_width, height)
            assert controller.stereo_difference[name] > 1.0
        print("PASS: scene and all three stereo video tracks deliver frames")

        assert await rpc(controller.room, robot, "acquire_control") == "true"
        await viewer.room.connect(url, viewer_token)
        assert await rpc(viewer.room, robot, "acquire_control") == "false"
        print("PASS: one controller holds the platform lease")

        initial_left = await controller.wait_joint("left", lambda _value: True)
        initial_right = await controller.wait_joint("right", lambda _value: True)
        left_target = -0.35 if initial_left[0] > 0 else 0.35
        left_track = await controller.room.local_participant.publish_data_track(
            name=config["arms"][0]["data_tracks"]["joint_control"])
        await asyncio.sleep(0.5)
        left_track.try_push(rtc.DataTrackFrame(payload=joint_command(robot, 1, left_target)))
        left = await controller.wait_joint("left", lambda value: abs(value[0] - left_target) < 0.04)
        right = await controller.wait_joint("right", lambda value: True)
        assert abs(left[0] - initial_left[0]) > 0.15
        assert abs(right[0] - initial_right[0]) < 0.05
        print("PASS: left arm moves while right arm holds position")

        right_target = -0.25 if initial_right[0] > 0 else 0.25
        right_track = await controller.room.local_participant.publish_data_track(
            name=config["arms"][1]["data_tracks"]["joint_control"])
        await asyncio.sleep(0.5)
        right_track.try_push(rtc.DataTrackFrame(payload=joint_command(robot, 1, right_target)))
        right = await controller.wait_joint("right", lambda value: abs(value[0] - right_target) < 0.04)
        assert abs(right[0] - initial_right[0]) > 0.15
        print("PASS: selected arm can switch and move independently")

        await asyncio.sleep(0.3)
        while not controller.telemetry.empty():
            controller.telemetry.get_nowait()
        right = await controller.wait_joint("right", lambda _value: True)
        pose_track = await controller.room.local_participant.publish_data_track(
            name=config["arms"][1]["data_tracks"]["pose"])
        await asyncio.sleep(0.5)
        # Engage with a zero displacement first, as the controller app does.
        # The arm then moves by the change from this first frame.
        pose_track.try_push(rtc.DataTrackFrame(payload=motion_command(
            robot, 1, (0, 0, 0, 0, 0, 0))))
        await asyncio.sleep(0.08)
        for sequence in range(2, 8):
            pose_track.try_push(rtc.DataTrackFrame(payload=motion_command(
                robot, sequence, (0.03, 0, 0, 0, 0, 0))))
            await asyncio.sleep(0.08)
        moved = await controller.wait_joint("right", lambda value: max(
            abs(value[index] - right[index]) for index in range(6)) > 0.01)
        assert moved != right
        pose_track.try_push(rtc.DataTrackFrame(payload=motion_command(
            robot, 8, (0, 0, 0, 0, 0, 0), active=False)))
        print("PASS: VR grip pose moves the selected arm")

        gripper_target = 0.0 if left[6] > 0.02 else 0.044
        left_track.try_push(rtc.DataTrackFrame(payload=joint_command(
            robot, 2, gripper_target, joint_index=6)))
        await controller.wait_joint("left", lambda value: abs(value[6] - gripper_target) < 0.006)
        print("PASS: gripper position moves the simulated gripper")

        await asyncio.sleep(0.3)
        while not controller.telemetry.empty():
            controller.telemetry.get_nowait()
        right = await controller.wait_joint("right", lambda _value: True)
        direction = -1.0 if right[0] > 0 else 1.0
        for sequence in range(2, 8):
            right_track.try_push(rtc.DataTrackFrame(payload=joint_command(
                robot, sequence, direction, kind=2)))
            await asyncio.sleep(0.08)
        await controller.wait_joint("right", lambda value: (value[0] - right[0]) * direction > 0.05)
        right_track.try_push(rtc.DataTrackFrame(payload=joint_command(robot, 8, 0.0, kind=2)))
        print("PASS: held VR joint velocity moves the selected arm")

        # The thumbstick sends gripper velocity, a signed fraction of the safe
        # speed. It runs beside grip pose, so send it during a pose stream.
        await asyncio.sleep(0.3)
        while not controller.telemetry.empty():
            controller.telemetry.get_nowait()
        right = await controller.wait_joint("right", lambda _value: True)
        gripper_direction = -1.0 if right[6] > 0.02 else 1.0
        for sequence in range(9, 17):
            pose_track.try_push(rtc.DataTrackFrame(payload=motion_command(robot, sequence, (0, 0, 0, 0, 0, 0))))
            right_track.try_push(rtc.DataTrackFrame(payload=joint_command(
                robot, sequence, gripper_direction, joint_index=6, kind=2)))
            await asyncio.sleep(0.08)
        await controller.wait_joint("right", lambda value: (value[6] - right[6]) * gripper_direction > 0.003)
        right_track.try_push(rtc.DataTrackFrame(payload=joint_command(robot, 17, 0.0, joint_index=6, kind=2)))
        pose_track.try_push(rtc.DataTrackFrame(payload=motion_command(
            robot, 17, (0, 0, 0, 0, 0, 0), active=False)))
        print("PASS: thumbstick gripper velocity moves the gripper during grip pose")

        viewer_track = await viewer.room.local_participant.publish_data_track(
            name=config["arms"][0]["data_tracks"]["joint_control"])
        await asyncio.sleep(0.5)
        while not controller.telemetry.empty():
            controller.telemetry.get_nowait()
        viewer_track.try_push(rtc.DataTrackFrame(payload=joint_command(robot, 1, -left_target)))
        await asyncio.sleep(0.8)
        left_after = None
        while not controller.telemetry.empty():
            arm, values = controller.telemetry.get_nowait()
            if arm == "left":
                left_after = values
        assert left_after is not None
        assert abs(left_after[0] - left_target) < 0.05, (
            f"unleased viewer moved the arm to {left_after[0]:.3f}")
        print("PASS: commands from the unleased viewer are ignored")

        await controller.close()
        for _ in range(30):
            if await rpc(viewer.room, robot, "acquire_control") == "true":
                break
            await asyncio.sleep(0.1)
        else:
            raise AssertionError("viewer could not acquire control after controller disconnect")
        print("PASS: control transfers after the controller leaves")
    finally:
        await controller.close()
        await viewer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("LIVEKIT_URL"), required=False)
    parser.add_argument("--robot", required=True,
                        help="Identity of the simulation participant, from its log")
    parser.add_argument("--controller-token", default=os.environ.get("CONTROLLER_TOKEN"))
    parser.add_argument("--viewer-token", default=os.environ.get("VIEWER_TOKEN"))
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--scene-width", type=int, default=960)
    parser.add_argument("--scene-height", type=int, default=720)
    args = parser.parse_args()
    if not args.url or not args.controller_token or not args.viewer_token:
        parser.error("LIVEKIT_URL, CONTROLLER_TOKEN, and VIEWER_TOKEN are required")
    try:
        asyncio.run(probe(args.url, args.robot, args.controller_token, args.viewer_token,
                          args.height, args.scene_width, args.scene_height))
    finally:
        gc.collect()


if __name__ == "__main__":
    main()
