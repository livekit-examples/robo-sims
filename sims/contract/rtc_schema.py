"""Schema bridge for pinned livekit==1.2.1.

The wheel's native/protobuf API supports schemas but its Python convenience
wrappers discard them. Keep all use of SDK internals here; remove this module
when the public Python API exposes these fields. No missing-metadata fallback.
"""

from __future__ import annotations

import asyncio
import json
from importlib.metadata import version

from livekit import rtc
from livekit.rtc._ffi_client import FfiClient
from livekit.rtc._proto import data_track_pb2 as dt, ffi_pb2 as ffi


def require_sdk():
    if version("livekit") != "1.2.1":
        raise RuntimeError("contract schema bridge requires livekit==1.2.1; install requirements.txt")


def schema_id(name: str):
    return dt.DataTrackSchemaId(name=name, encoding=dt.DataTrackSchemaEncoding(
        well_known=dt.DataTrackSchemaEncoding.WELL_KNOWN_SCHEMA_ENCODING_JSON_SCHEMA))


class SchemaRoom(rtc.Room):
    """Retain wire publication metadata omitted by the pinned Python wrapper."""

    def __init__(self):
        require_sdk()
        super().__init__()

    def _on_room_event(self, event):
        if event.WhichOneof("message") == "data_track_published":
            owned = event.data_track_published.track
            track = rtc.RemoteDataTrack(owned)
            track.info.schema = dt.DataTrackSchemaId()
            track.info.schema.CopyFrom(owned.info.schema)
            track.info.frame_encoding = dt.DataTrackFrameEncoding()
            track.info.frame_encoding.CopyFrom(owned.info.frame_encoding)
            self.emit("data_track_published", track)
        else:
            super()._on_room_event(event)


def matches_json(track, msg_type: str) -> bool:
    schema = getattr(track.info, "schema", None)
    encoding = getattr(track.info, "frame_encoding", None)
    return (schema is not None and encoding is not None and schema == schema_id(msg_type)
            and encoding.well_known == dt.DataTrackFrameEncoding.WELL_KNOWN_FRAME_ENCODING_JSON)


async def _request(request, field):
    queue = FfiClient.instance.queue.subscribe()
    try:
        response = FfiClient.instance.request(request)
        async_id = getattr(response, field).async_id
        event = await asyncio.wait_for(queue.wait_for(
            lambda e: e.WhichOneof("message") == field and getattr(e, field).async_id == async_id), 10)
        callback = getattr(event, field)
        if callback.HasField("error"):
            raise RuntimeError(str(callback.error))
        return callback
    finally:
        FfiClient.instance.queue.unsubscribe(queue)


async def publish_json(participant, name: str, msg_type: str, definition: dict):
    require_sdk()
    request = ffi.FfiRequest()
    request.define_schema.local_participant_handle = participant._ffi_handle.handle
    request.define_schema.schema_id.CopyFrom(schema_id(msg_type))
    request.define_schema.definition = json.dumps(definition, allow_nan=False)
    await _request(request, "define_schema")
    request = ffi.FfiRequest()
    request.publish_data_track.local_participant_handle = participant._ffi_handle.handle
    options = request.publish_data_track.options
    options.name = name
    options.schema.CopyFrom(schema_id(msg_type))
    options.frame_encoding.well_known = dt.DataTrackFrameEncoding.WELL_KNOWN_FRAME_ENCODING_JSON
    result = await _request(request, "publish_data_track")
    return rtc.LocalDataTrack(result.track)
