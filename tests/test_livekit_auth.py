"""Identity generation must work in a standalone robo-sims checkout."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sims.livekit_auth import ServerSettings, new_identity, scientist_names
from sims.paths import SCIENTIST_NAMES_FILE, SIMULATION_DIR


class IdentityTests(unittest.IsolatedAsyncioTestCase):
    async def test_identity_uses_bundled_names_and_skips_taken_names(self):
        self.assertTrue(SCIENTIST_NAMES_FILE.is_relative_to(SIMULATION_DIR))
        names = scientist_names()
        self.assertTrue(names)
        self.assertEqual(len(names), len(set(names)))
        prefix = "simulation-trossen-workbench-"
        participants = [SimpleNamespace(identity=prefix + name) for name in names[:-1]]
        client = AsyncMock()
        client.room.list_participants.return_value = SimpleNamespace(participants=participants)
        with patch("sims.livekit_auth.api.LiveKitAPI") as api:
            api.return_value.__aenter__.return_value = client
            identity = await new_identity(
                ServerSettings("ws://localhost:7880", "key", "secret", "room"),
                "trossen-workbench",
            )
        self.assertEqual(identity, prefix + names[-1])
