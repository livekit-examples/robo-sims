"""Session isolation and recovery without sockets or native simulation imports."""
import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from aiohttp import web
from sims.demo.files import write_json
from sims.demo.supervisor import Supervisor, now


class Provider:
    def __init__(self):
        self.alive = set()
    async def start(self, run, *_):
        self.alive.add(run['id'])
        return 42
    def status(self, run):
        return run['id'] in self.alive
    def stop(self, run):
        self.alive.discard(run['id'])


class DemoTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.provider = Provider()
        self.s = Supervisor(Path(self.directory.name), 'k' * 32, 'ws://localhost:17880',
                            'devkey', 'secret', provider=self.provider)
        self.s.lk = type('LK', (), {'room': type('Rooms', (), {
            'create_room': AsyncMock(), 'delete_room': AsyncMock()})()})()
        self.s.launches = set()
        self.catalog = patch.object(self.s, 'catalog', return_value=[{'id': 'trossen-wxai', 'available': True}])
        self.catalog.start()
    async def asyncTearDown(self):
        await asyncio.gather(*self.s.launches, return_exceptions=True)
        self.catalog.stop()
        self.s.db.close(); self.s.lock.close(); self.directory.cleanup()
    async def launch(self, owner='owner-aaaaaaaaaaaa', key='request-aaaaaaaaaa'):
        run = await self.s.launch(owner, 'trossen-wxai', key)
        await asyncio.gather(*self.s.launches)
        return self.s.owned(run['id'], owner)

    async def test_retry_does_not_spawn_again_and_owner_cannot_access_other_run(self):
        run = await self.launch()
        again = await self.launch()
        self.assertEqual(run['id'], again['id'])
        self.assertEqual(self.s.lk.room.create_room.await_count, 1)
        with self.assertRaises(web.HTTPNotFound):
            self.s.owned(run['id'], 'another-owner-aaaa')
        with self.assertRaises(web.HTTPConflict):
            await self.launch(key='request-bbbbbbbbbb')

    async def test_capacity_reserved_before_start_and_one_failure_leaves_other_run(self):
        first = await self.launch()
        second = await self.launch('owner-bbbbbbbbbbbb')
        with self.assertRaises(web.HTTPTooManyRequests):
            await self.launch('owner-cccccccccccc')
        self.s.begin_stop(first, 'failed')
        first = self.s.owned(first['id'], first['owner'])
        first['stop_at'] = now() - 10000; self.s.save(first)
        await self.s.tick()
        self.assertEqual(self.s.owned(first['id'], first['owner'])['state'], 'failed')
        self.assertTrue(self.provider.status(second))
        self.assertNotEqual(first['room'], second['room'])

    async def test_worker_readiness_sets_fixed_deadline_and_expiry_cleans_room(self):
        run = await self.launch()
        event = dict(state='ready', ready_at=now(), expires_at=now() + 60000, policy={})
        write_json(Path(self.directory.name) / run['id'] / 'event.json', event)
        await self.s.tick()
        ready = self.s.owned(run['id'], run['owner'])
        self.assertEqual(ready['expires_at'], event['expires_at'])
        await self.s.tick()
        self.assertEqual(self.s.owned(run['id'], run['owner'])['expires_at'], event['expires_at'])
        with patch('sims.demo.supervisor.now', return_value=event['expires_at'] + 10000):
            await self.s.tick()
        with patch('sims.demo.supervisor.now', return_value=event['expires_at'] + 20000):
            await self.s.tick()
        self.assertEqual(self.s.owned(run['id'], run['owner'])['state'], 'expired')
        self.assertFalse(self.provider.status(run))
        self.s.lk.room.delete_room.assert_awaited()

    async def test_stop_during_start_is_not_overwritten_by_late_process_handle(self):
        gate = asyncio.Event()
        original = self.provider.start
        async def slow(*args):
            await gate.wait()
            return await original(*args)
        self.provider.start = slow
        run = await self.s.launch('owner-aaaaaaaaaaaa', 'trossen-wxai', 'request-aaaaaaaaaa')
        await asyncio.sleep(0)
        self.s.begin_stop(run)
        gate.set()
        await asyncio.gather(*self.s.launches)
        self.assertEqual(self.s.owned(run['id'], run['owner'])['state'], 'stopping')
