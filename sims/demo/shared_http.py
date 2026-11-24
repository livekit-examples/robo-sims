"""One HTTP contract for local and hosted shared simulations."""
import re
from .shared_rules import Problem

ID = r'[a-f0-9]{32}'


def resolve(method, path, query, body, proof, owner):
    if not isinstance(body, dict):
        raise Problem(400, 'Invalid request.')
    if method == 'GET' and path == 'simulations':
        return 'simulations', ()
    match = re.fullmatch(r'simulations/([a-z0-9-]+)/watch', path)
    if method == 'POST' and match:
        return 'watch', (owner, match[1])
    match = re.fullmatch(r'instances/('+ID+r')', path)
    if method == 'GET' and match:
        return 'instance', (match[1],)
    if method == 'POST' and path == 'credentials':
        return 'room_credentials', (owner,)
    match = re.fullmatch(r'instances/('+ID+r')/credentials', path)
    if method == 'POST' and match:
        return 'credentials', (match[1], owner, {**body, 'proof': proof})
    if path == 'queue' and method == 'GET':
        return 'queue', (query.get('instance', ''), owner, proof)
    if path in ('queue', 'engagement') and method == 'POST':
        identifier = body.get('instance_id', '')
        if not re.fullmatch(ID, identifier):
            raise Problem(400, 'Invalid simulation instance.')
        if path == 'queue':
            if body.get('action') not in ('join', 'leave', 'release'):
                raise Problem(400, 'Choose join, leave, or release.')
            return 'queue', (identifier, owner, proof, body['action'], bool(body.get('resume')))
        return 'engagement', (identifier, owner, {**body, 'proof': proof})
    raise Problem(404, 'Unknown simulation route.')
