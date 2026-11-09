"""Durable Modal snapshot contract; no network or Modal account needed."""
import copy
import unittest
from sims.demo.hosted_store import ModalStore


class Dictionary:
    def __init__(self): self.data={}
    def get(self,key,default=None): return copy.deepcopy(self.data.get(key,default))
    def put(self,key,value): self.data[key]=copy.deepcopy(value)


class StoreTests(unittest.TestCase):
    def test_reader_cannot_mutate_and_replacement_keeps_state(self):
        dictionary=Dictionary()
        store=ModalStore(dictionary,'sessions',writable=True)
        with store.transaction() as tx:
            tx.save(dict(id='a',state='queued'))
            self.assertTrue(tx.reserve('curie-00','a',100))
        reader=ModalStore(dictionary,'sessions')
        self.assertEqual(reader.records(),[dict(id='a',state='queued')])
        with self.assertRaises(RuntimeError):
            with reader.transaction(): pass
        with store.transaction() as tx:
            self.assertFalse(tx.reserve('curie-00','b',200))
        with self.assertRaises(ValueError):
            with store.transaction() as tx:
                tx.save(dict(id='b'))
                raise ValueError('failed mutation')
        self.assertEqual(len(reader.records()),1)

    def test_reservation_retained_until_cleanup_and_token_expiry(self):
        store=ModalStore(Dictionary(),'sessions',writable=True)
        run=dict(id='a',state='stopped',room_name='curie-00',cleaned_at=1,token_until=100)
        with store.transaction() as tx:
            tx.save(run); tx.reserve('curie-00','a',100)
            tx.prune(20,50)
            self.assertFalse(tx.reserve('curie-00','b',200))
        with store.transaction() as tx:
            tx.prune(20,101)
            self.assertIsNone(tx.get('a'))
            self.assertTrue(tx.reserve('curie-00','b',200))
