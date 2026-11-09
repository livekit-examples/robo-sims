"""Modal-only durable state. All writes run in the serialized coordinator Function.

Dict does not supply SQL transactions. A transaction here is a single snapshot
write; callers MUST route mutations through coordinate (one container, one input).
Readers can access the last committed snapshot from any API container.
"""
from contextlib import contextmanager
import copy


class ModalStore:
    def __init__(self, dictionary, namespace, writable=False):
        self.dictionary, self.namespace, self.writable = dictionary, namespace, writable

    def snapshot(self):
        return self.dictionary.get(self.namespace, {'runs': {}, 'rooms': {}})

    def records(self):
        return list(self.snapshot()['runs'].values())

    @contextmanager
    def transaction(self):
        if not self.writable:
            raise RuntimeError('Mutations must run through the serialized Modal coordinator.')
        snapshot = copy.deepcopy(self.snapshot())
        yield Snapshot(snapshot)
        self.dictionary.put(self.namespace, snapshot)


class Snapshot:
    def __init__(self, snapshot):
        self.snapshot = snapshot

    def records(self):
        return list(self.snapshot['runs'].values())

    def get(self, identifier):
        return self.snapshot['runs'].get(identifier)

    def save(self, run):
        self.snapshot['runs'][run['id']] = run

    def reserve(self, name, identifier, until):
        if name in self.snapshot['rooms']:
            return False
        self.snapshot['rooms'][name] = dict(session=identifier, until=until)
        return True

    def prune(self, before, timestamp):
        for run in list(self.records()):
            if run.get('cleaned_at') and run['token_until'] < timestamp:
                name = run['room_name']
                if self.snapshot['rooms'].get(name, {}).get('session') == run['id']:
                    del self.snapshot['rooms'][name]
                if run['cleaned_at'] < before:
                    del self.snapshot['runs'][run['id']]
