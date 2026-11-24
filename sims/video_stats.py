"""Sender-side camera timing, before encoding and network delivery."""

from collections import defaultdict
import time


class VideoStats:
    def __init__(self, fps, logger, interval=5.0):
        self.fps, self.logger, self.interval = fps, logger, interval
        self.started = time.perf_counter()
        self.cycles = []
        self.cameras = defaultdict(list)

    def camera(self, key, prepare, render, capture):
        self.cameras[key].append((prepare, render, capture))

    def cycle(self, started):
        now = time.perf_counter()
        self.cycles.append(now - started)
        if now - self.started >= self.interval:
            self.report(now)

    def report(self, now=None):
        if not self.cycles:
            return
        now = time.perf_counter() if now is None else now
        elapsed = now - self.started
        count = len(self.cycles)
        mean = sum(self.cycles) / count
        p95 = sorted(self.cycles)[min(count - 1, int(count * .95))]
        late = sum(value > 1 / self.fps for value in self.cycles)
        self.logger.info(
            "Video source: target=%g fps, submitted=%.1f fps/camera, "
            "work=%.1f ms/cycle (p95=%.1f), budget=%.1f ms, over-budget=%d/%d",
            self.fps, count / elapsed, mean * 1000, p95 * 1000,
            1000 / self.fps, late, count)
        for key, values in self.cameras.items():
            means = [1000 * sum(row[i] for row in values) / len(values) for i in range(3)]
            self.logger.info("Video camera %s: prepare=%.2f ms, render/wait=%.2f ms, capture=%.2f ms",
                             key, *means)
        self.started = now
        self.cycles.clear()
        self.cameras.clear()
