"""LiveKit owns queue state. Call mutations only inside the single writer."""
import json
import math
import uuid

from aiohttp import ClientTimeout
from livekit import api

from .shared_rules import (ACTIVITY, CONTROL_ROBOT, GENERATION, KIND, QUEUED,
                           QUEUE_ROBOT, UNTIL, Problem, activity, engaged, timestamp)
from .rules import now


def missing(error):
    return getattr(error, 'code', '') == 'not_found'


class LiveRoom:
    def __init__(self, env, name):
        self.env, self.name = env, name

    async def __aenter__(self):
        self.client = api.LiveKitAPI(self.env['LIVEKIT_URL'], self.env['LIVEKIT_API_KEY'],
            self.env['LIVEKIT_API_SECRET'], timeout=ClientTimeout(total=4))
        return self

    async def __aexit__(self, *_):
        await self.client.aclose()

    async def participants(self):
        try:
            return list((await self.client.room.list_participants(
                api.ListParticipantsRequest(room=self.name))).participants)
        except Exception as error:
            if missing(error):
                return []
            raise

    async def update(self, identity, attributes, publishing=None):
        request = api.UpdateParticipantRequest(room=self.name, identity=identity, attributes=attributes)
        if publishing is not None:
            request.permission.CopyFrom(api.ParticipantPermission(can_subscribe=True,
                can_publish=publishing, can_publish_data=publishing, can_update_metadata=False))
        try:
            return await self.client.room.update_participant(request)
        except Exception as error:
            if missing(error):
                return None
            raise

    async def remove(self, identity):
        try:
            await self.client.room.remove_participant(api.RoomParticipantIdentity(room=self.name, identity=identity))
        except Exception as error:
            if not missing(error):
                raise

    async def ensure(self):
        # Room metadata belongs to the service, never to an individual worker.
        await self.client.room.create_room(api.CreateRoomRequest(name=self.name, empty_timeout=90))


class ControlQueue:
    def __init__(self, room, run):
        self.room, self.run = room, run
        self.robot = run['robot_identity']

    async def viewer(self, identity):
        peer = next((p for p in await self.room.participants() if p.identity == identity), None)
        if peer is None or peer.attributes.get(KIND) != 'viewer':
            raise Problem(409, 'Connect to the shared room before requesting control.')
        return peer

    async def engage(self, identity, event_id, *, worker=False):
        peer = await self.viewer(identity)
        values = activity(peer)
        old = values.get(self.run['id'], {})
        if worker and (not self.controls(peer, now()) or not engaged(peer, self.run, now())):
            return
        events = old.get('events', [old.get('event')])
        if event_id not in events:
            # Retry of the current event cannot turn a dead tab into new activity.
            values[self.run['id']] = dict(generation=self.run['id'], at=now(), event=event_id,
                                         events=(events+[event_id])[-64:])
            await self.room.update(identity, {ACTIVITY: json.dumps(values, separators=(',', ':'))})
        return values[self.run['id']]['at']

    def controls(self, peer, at):
        return (peer.attributes.get(CONTROL_ROBOT) == self.robot and
                peer.attributes.get(GENERATION) == self.run['id'] and timestamp(peer, UNTIL) > at)

    async def revoke(self, peer, requeue=None):
        attributes = {UNTIL: '', CONTROL_ROBOT: '', GENERATION: ''}
        if requeue is not None:
            attributes.update({QUEUED: str(requeue), QUEUE_ROBOT: self.robot})
        # A failed update propagates: never grant someone else after uncertain revocation.
        await self.room.update(peer.identity, attributes, publishing=False)
        peer.attributes.update(attributes)

    async def action(self, identity, action, *, engage_join=True):
        peer = await self.viewer(identity)
        if action == 'join':
            target = peer.attributes.get(CONTROL_ROBOT) or peer.attributes.get(QUEUE_ROBOT)
            if target and target != self.robot:
                raise Problem(409, 'Leave your current queue or release control before switching simulations.')
            if not engaged(peer, self.run, now()):
                raise Problem(410, 'Continue watching before joining the queue.')
            if not target:
                peers = await self.room.participants()
                if sum(p.attributes.get(QUEUE_ROBOT) == self.robot for p in peers) >= 200:
                    raise Problem(429, 'This control queue is full.')
                if engage_join:
                    await self.engage(identity, 'join-'+uuid.uuid4().hex)
                await self.room.update(identity, {QUEUED: str(now()), QUEUE_ROBOT: self.robot})
        elif action in ('leave', 'release'):
            if peer.attributes.get(CONTROL_ROBOT) == self.robot:
                await self.revoke(peer)
            if peer.attributes.get(QUEUE_ROBOT) == self.robot:
                await self.room.update(identity, {QUEUED: '', QUEUE_ROBOT: ''})
        else:
            raise Problem(400, 'Choose join, leave, or release.')
        return await self.reconcile(identity)

    async def reconcile(self, identity=None, *, stopping=False):
        at = now()
        peers = await self.room.participants()
        robot_online = any(p.identity == self.robot for p in peers)
        viewers = [p for p in peers if p.attributes.get(KIND) == 'viewer']
        removed = set()
        for peer in viewers:
            eligible = not stopping and engaged(peer, self.run, at)
            if peer.attributes.get(CONTROL_ROBOT) == self.robot and not eligible:
                await self.revoke(peer)
            if peer.attributes.get(QUEUE_ROBOT) == self.robot and not eligible:
                await self.room.update(peer.identity, {QUEUED: '', QUEUE_ROBOT: ''})
                peer.attributes.update({QUEUED: '', QUEUE_ROBOT: ''})
            values = activity(peer)
            if self.run['id'] in values and not eligible:
                del values[self.run['id']]
                await self.room.update(peer.identity, {ACTIVITY: json.dumps(values, separators=(',', ':'))})
                peer.attributes[ACTIVITY] = json.dumps(values, separators=(',', ':'))
                if not values:
                    await self.room.remove(peer.identity)
                    removed.add(peer.identity)
        peers = [p for p in peers if p.identity not in removed]
        queue = sorted((p for p in peers if p.attributes.get(QUEUE_ROBOT) == self.robot and
                        timestamp(p, QUEUED) and engaged(p, self.run, at)),
                       key=lambda p: (timestamp(p, QUEUED), p.identity))
        holders = sorted((p for p in peers if p.attributes.get(CONTROL_ROBOT) == self.robot),
                         key=lambda p: (timestamp(p, UNTIL), p.identity))
        controller = None
        for peer in holders:
            valid = (robot_online and not stopping and engaged(peer, self.run, at) and
                     peer.attributes.get(GENERATION) == self.run['id'])
            if controller is None and valid and timestamp(peer, UNTIL) > at:
                controller = dict(identity=peer.identity, until=timestamp(peer, UNTIL))
            elif controller is None and valid and not queue and at + self.run['turn_seconds']*1000 < self.run['expires_at']:
                until = at + self.run['turn_seconds'] * 1000
                await self.room.update(peer.identity, {UNTIL: str(until)})
                controller = dict(identity=peer.identity, until=until)
            else:
                await self.revoke(peer)
        if controller is None and queue and robot_online and not stopping and at < self.run['expires_at']:
            while queue:
                peer = queue.pop(0)
                until = min(at + self.run['turn_seconds'] * 1000, self.run['expires_at'])
                result = await self.room.update(peer.identity, {QUEUED: '', QUEUE_ROBOT: '',
                    CONTROL_ROBOT: self.robot, GENERATION: self.run['id'], UNTIL: str(until)}, publishing=True)
                if result is not None:
                    controller = dict(identity=peer.identity, until=until)
                    break
        ids = [p.identity for p in queue]
        position = ids.index(identity) + 1 if identity in ids else None
        remaining = max(0, controller['until'] - at) / 1000 if controller else 0
        mine = next((p for p in peers if p.identity == identity), None)
        value = activity(mine).get(self.run['id'], {}) if mine else {}
        return dict(instance_id=self.run['id'], robot_identity=self.robot, controller=controller,
            queue=ids, turn_seconds=self.run['turn_seconds'], server_time=at, queue_position=position,
            estimated_wait_seconds=math.ceil(remaining + (position-1)*self.run['turn_seconds']) if position else None,
            queued_at=timestamp(mine, QUEUED) if mine and position else None,
            engaged_until=value.get('at', 0) + self.run['idle_seconds']*1000 if value else None,
            viewer_count=sum(engaged(p, self.run, at) for p in peers))
