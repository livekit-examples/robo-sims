"""LiveKit credentials of a simulated robot.

A simulation creates its own token from the API key and secret of the
controller app, in controller/.env.local of the teleop repository, with a
free identity simulation-<sim>-<scientist>, picked like the
controller-<scientist> identities of the controller app. A token given with
--token or LIVEKIT_TOKEN is used as is instead.

For a hardware-level simulation, the token is handed to the robot program.
The API secret never leaves this process.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

from livekit import api

from .paths import SCIENTIST_NAMES_FILE

IDENTITY_ATTEMPTS = 20


@dataclass(frozen=True)
class ServerSettings:
    url: str
    api_key: str
    api_secret: str
    room: str


@dataclass(frozen=True)
class Credentials:
    url: str
    token: str
    identity: str | None  # None for a token from --token or LIVEKIT_TOKEN
    room: str | None


def read_env_file(path: Path) -> dict[str, str]:
    """Reads KEY=VALUE lines. Blank lines and lines that start with # are skipped."""
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip().removeprefix("export ").strip()] = value
    return values


def server_settings(env_file: Path) -> ServerSettings:
    """Takes each setting from the process environment, or else from env_file."""
    file_values = read_env_file(env_file) if env_file.is_file() else {}
    names = {"url": "LIVEKIT_URL", "api_key": "LIVEKIT_API_KEY",
             "api_secret": "LIVEKIT_API_SECRET", "room": "LIVEKIT_ROOM_NAME"}
    values = {field: os.environ.get(name) or file_values.get(name, "") for field, name in names.items()}
    missing = [names[field] for field, value in values.items() if not value]
    if missing:
        raise ValueError(f"Set {', '.join(missing)} in {env_file} or in the environment")
    return ServerSettings(**values)


def scientist_names() -> list[str]:
    return json.loads(SCIENTIST_NAMES_FILE.read_text(encoding="utf-8"))


async def new_identity(settings: ServerSettings, sim_name: str) -> str:
    """Picks simulation-<sim>-<scientist> that no participant in the room uses.

    Two participants with one identity cannot share a room: LiveKit removes
    the older one. If every name is in use, a number goes after a random name.
    """
    prefix = f"simulation-{sim_name.replace('_', '-')}-"
    taken: set[str] = set()
    try:
        async with api.LiveKitAPI(settings.url, settings.api_key, settings.api_secret) as lkapi:
            response = await lkapi.room.list_participants(api.ListParticipantsRequest(room=settings.room))
            taken = {participant.identity for participant in response.participants}
    except Exception:
        pass  # The room does not exist yet, or the lookup failed. Any name works.
    names = scientist_names()
    free = [f"{prefix}{name}" for name in names if f"{prefix}{name}" not in taken]
    if free:
        return secrets.choice(free)
    identity = f"{prefix}{secrets.choice(names)}-{secrets.randbelow(98) + 2}"
    for _ in range(IDENTITY_ATTEMPTS - 1):
        if identity not in taken:
            break
        identity = f"{prefix}{secrets.choice(names)}-{secrets.randbelow(98) + 2}"
    return identity


def create_token(settings: ServerSettings, identity: str) -> str:
    """A token that lets the robot join the room, publish, subscribe and serve RPCs."""
    grants = api.VideoGrants(room_join=True, room=settings.room, can_publish=True,
                             can_subscribe=True, can_publish_data=True)
    return (api.AccessToken(settings.api_key, settings.api_secret)
            .with_identity(identity)
            .with_grants(grants)
            .to_jwt())


async def credentials(args: argparse.Namespace, sim_name: str) -> Credentials:
    """The URL and token to join with, from --token/LIVEKIT_TOKEN or else a new token."""
    token = args.token or os.environ.get("LIVEKIT_TOKEN")
    if token:
        url = args.url or os.environ.get("LIVEKIT_URL")
        if not url and args.env_file.is_file():
            url = read_env_file(args.env_file).get("LIVEKIT_URL")
        if not url:
            raise SystemExit("LIVEKIT_URL (or --url) is required with --token")
        return Credentials(url, token, None, None)
    try:
        settings = server_settings(args.env_file)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    identity = await new_identity(settings, sim_name)
    return Credentials(args.url or settings.url, create_token(settings, identity), identity, settings.room)
