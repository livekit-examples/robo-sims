"""Verify deployed pool wiring and recovery behavior without invoking Modal."""
import ast
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from test_demo_hosted import MemoryStore
from sims.demo.hosted import HostedService
from sims.demo.rules import SUPPORTED, now


class ModalTests(unittest.TestCase):
    def test_all_simulations_use_one_gpu_function_with_twenty_single_input_containers(self):
        source=ast.parse((Path(__file__).resolve().parents[1]/'modal_app.py').read_text())
        gpu_functions=[]
        for function in (node for node in source.body if isinstance(node,ast.FunctionDef)):
            for decorator in function.decorator_list:
                if not isinstance(decorator,ast.Call): continue
                keywords={kw.arg:kw.value for kw in decorator.keywords}
                if 'gpu' in keywords:
                    gpu_functions.append(function.name)
                    self.assertEqual(ast.literal_eval(keywords['max_containers']),20)
                    self.assertFalse(any(isinstance(d,ast.Call) and isinstance(d.func,ast.Attribute)
                        and d.func.attr=='concurrent' for d in function.decorator_list))
                if function.name=='coordinate' and 'max_containers' in keywords:
                    self.assertEqual(ast.literal_eval(keywords['max_containers']),1)
        self.assertEqual(gpu_functions,['run_simulation'])

    def test_unknown_poll_does_not_free_a_running_slot(self):
        import modal_app
        store=MemoryStore()
        svc=HostedService(store,dict(SIM_SUPERVISOR_KEY='k'*32,SIM_IMAGE_VERSION='test'),
                          [dict(id=s,available=True) for s in SUPPORTED])
        run=svc.launch('owner-aaaaaaaaaaaa',dict(simulation_id=SUPPORTED[0],request_id='request-aaaaaaaaaa'))
        record=store.get(run['id'])
        record.update(state='ready',call_id='fc-test',expires_at=now()+60000,start_deadline=now()+30000)
        store.save(record)
        provider=Mock()
        call=Mock()
        call.get.side_effect=ConnectionError('network unavailable')
        with patch.object(modal_app,'service',return_value=svc), \
                patch.object(modal_app,'command') as command, \
                patch.object(modal_app,'ModalProvider',return_value=provider), \
                patch.object(modal_app.modal.FunctionCall,'from_id',return_value=call), \
                patch('time.monotonic',side_effect=[0,100]):
            modal_app.sweep.local()
        provider.delete_room.assert_not_called()
        call.cancel.assert_not_called()
        self.assertFalse(any(c.args[0]=='finish' for c in command.call_args_list))
