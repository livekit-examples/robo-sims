"""Exercise the hosted HTTP contract without allocating cloud compute."""
import asyncio
import importlib.util
import unittest
from unittest.mock import patch
from sims.demo.hosted_store import ModalStore
from sims.demo.shared import SharedService
from test_shared import Dictionary, Room


@unittest.skipUnless(importlib.util.find_spec('fastapi'), 'install requirements-modal.txt')
class HttpTests(unittest.TestCase):
    def test_private_auth_public_instances_and_no_worker_or_stop_routes(self):
        from fastapi.testclient import TestClient
        import modal_app
        room=Room()
        svc=SharedService(ModalStore(Dictionary(),'test',writable=True),
            dict(SIM_SUPERVISOR_KEY='k'*32,LIVEKIT_URL='ws://local',LIVEKIT_API_KEY='key',LIVEKIT_API_SECRET='s'*32),
            [dict(id='vacuum',available=True)],lambda *_:room)
        with patch.object(modal_app,'service',return_value=svc):
            app=modal_app.api.local()
        def execute(action,*args): return asyncio.run(svc.handle(action,*args))
        headers={'Authorization':'Bearer '+'k'*32,'X-Demo-Owner':'owner-aaaaaaaaaaaa'}
        with patch.object(modal_app,'service',return_value=svc), TestClient(app) as client, \
                patch.object(modal_app,'command',side_effect=execute), patch.object(modal_app,'request_dispatch'):
            self.assertEqual(client.get('/simulations').status_code,401)
            self.assertEqual(client.get('/simulations',headers=headers).status_code,200)
            response=client.post('/simulations/vacuum/watch',headers=headers,json={})
            self.assertEqual(response.status_code,200,response.text)
            identifier=response.json()['id']
            other={**headers,'X-Demo-Owner':'owner-bbbbbbbbbbbb'}
            self.assertEqual(client.get('/instances/'+identifier,headers=other).status_code,200)
            self.assertEqual(client.post('/simulations/vacuum/watch',headers=other,json={}).json()['id'],identifier)
            self.assertEqual(client.delete('/instances/'+identifier,headers=headers).status_code,405)
            self.assertEqual(client.post('/workers/'+identifier+'/events',headers=headers,json={}).status_code,404)
            self.assertEqual(client.post('/queue',headers=headers,json=[]).status_code,400)
            credential=client.post('/credentials',headers=headers,json={}).json()
            self.assertIn('proof',credential)
            self.assertNotIn('SIM_SUPERVISOR_KEY',response.text)
            self.assertEqual(response.headers['cache-control'],'no-store')
