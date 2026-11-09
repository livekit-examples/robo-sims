"""Exercise Modal's actual ASGI factory without cloud compute or a database."""
import importlib.util
import unittest
from unittest.mock import patch

from test_demo_hosted import MemoryStore
from sims.demo.hosted import HostedService
from sims.demo.rules import SUPPORTED


@unittest.skipUnless(importlib.util.find_spec('fastapi'), 'install requirements-modal.txt')
class HttpTests(unittest.TestCase):
    def test_auth_launch_ownership_and_strict_input(self):
        from fastapi.testclient import TestClient
        import modal_app
        svc = HostedService(MemoryStore(), dict(SIM_SUPERVISOR_KEY='k'*32, SIM_IMAGE_VERSION='test'),
                            [dict(id=s, available=True) for s in SUPPORTED])
        with patch.object(modal_app, 'service', return_value=svc):
            app = modal_app.api.local()
        headers = {'Authorization':'Bearer '+'k'*32, 'X-Demo-Owner':'owner-aaaaaaaaaaaa'}
        with TestClient(app) as client, patch.object(modal_app, 'command', side_effect=lambda action,*args: svc.launch(*args) if action=='launch' else svc.report(*args)), patch.object(modal_app, 'request_dispatch') as spawn:
            self.assertEqual(client.get('/sessions').status_code,401)
            self.assertEqual(len(client.get('/simulations',headers=headers).json()),len(SUPPORTED))
            self.assertEqual(client.post('/sessions',headers=headers,json={'extra':True}).status_code,422)
            response = client.post('/sessions',headers=headers,json={
                'simulation_id':SUPPORTED[0], 'request_id':'request-aaaaaaaaaa'})
            self.assertEqual(response.status_code,202,response.text)
            identifier = response.json()['id']
            spawn.assert_called_once_with()
            self.assertEqual(client.get('/sessions/'+identifier, headers={**headers,
                'X-Demo-Owner':'another-owner-aaaa'}).status_code,404)
            self.assertEqual(client.post('/workers/'+identifier+'/events',json={'state':'ready'}).status_code,401)
            self.assertEqual(response.headers['cache-control'],'no-store')
