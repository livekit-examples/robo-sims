"""One FIFO across all scenes, with wait estimates derived from occupied slots."""
import heapq
import math

RUNNING = ('starting', 'ready', 'stopping')
PENDING = ('queued', *RUNNING)
HOSTED_CAPACITY = 20


def ordered(runs):
    return sorted((r for r in runs if r['state'] == 'queued'),
                  key=lambda r: (r.get('queue_sequence', r['created_at']), r['id']))


def queue_info(run, runs, capacity, timestamp):
    if run['state'] != 'queued':
        return dict(queue_position=None, queue_length=None, estimated_wait_seconds=None)
    waiting = ordered(runs)
    slots = []
    for active in (r for r in runs if r['state'] in RUNNING):
        cleanup = active.get('cleanup_seconds', 5)
        if active['state'] == 'ready':
            available = (active['expires_at'] - timestamp) / 1000 + cleanup
        elif active['state'] == 'stopping':
            available = (active['stop_at'] - timestamp) / 1000 + cleanup
            if available < -30:
                available = math.inf  # Cleanup/recovery is delayed; do not promise an immediate slot.
        else:
            started = active.get('started_at', active.get('admitted_at', active['created_at']))
            available = ((started - timestamp) / 1000 + active.get('startup_seconds', 45)
                         + 30 + active.get('duration_seconds', 60) + cleanup)
        slots.append(max(0, available))
    slots.extend([0] * max(0, capacity - len(slots)))
    heapq.heapify(slots)
    for index, candidate in enumerate(waiting, 1):
        available = heapq.heappop(slots)
        if candidate['id'] == run['id']:
            return dict(queue_position=index, queue_length=len(waiting),
                        estimated_wait_seconds=max(1, math.ceil(available)) if math.isfinite(available) else None)
        heapq.heappush(slots, available + candidate.get('startup_seconds', 45)
                       + candidate.get('duration_seconds', 60) + candidate.get('cleanup_seconds', 5))
    return dict(queue_position=None, queue_length=len(waiting), estimated_wait_seconds=None)
