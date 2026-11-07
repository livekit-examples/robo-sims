"""Native process provider. No simulation modules are imported here."""
import asyncio
import os
import signal
import sys
from pathlib import Path


class NativeProvider:
    def __init__(self):
        self.children = {}

    async def start(self, run, directory: Path, token: str, url: str, cleanup: float):
        allowed = ('PATH', 'HOME', 'TMPDIR', 'MUJOCO_GL', 'SIM_ASSETS_DIR', 'DYLD_LIBRARY_PATH',
                   'LD_LIBRARY_PATH', 'PYTHONPATH')
        env = {k: os.environ[k] for k in allowed if k in os.environ}
        env.update(LIVEKIT_URL=url, LIVEKIT_TOKEN=token, PYTHONUNBUFFERED='1')
        log = (directory / 'worker.log').open('ab')
        try:
            child = await asyncio.create_subprocess_exec(
                sys.executable, '-m', 'sims.demo.worker', '--id', run['id'],
                '--simulation', run['simulation_id'], '--directory', str(directory),
                '--cleanup', str(cleanup), cwd=Path(__file__).resolve().parents[2],
                env=env, stdout=log, stderr=log, start_new_session=True)
        finally:
            log.close()
        self.children[run['id']] = child
        return child.pid

    def status(self, run):
        child = self.children.get(run['id'])
        if child:
            return child.returncode is None
        # Recovery: match the run marker before touching a surviving process group.
        import subprocess
        try:
            command = subprocess.check_output(['ps', '-p', str(run['pid']), '-o', 'args='], text=True)
            return 'sims.demo.worker' in command and f"--id {run['id']} " in command
        except (KeyError, subprocess.SubprocessError):
            return False

    def stop(self, run):
        if self.status(run):
            try:
                os.killpg(run['pid'], signal.SIGKILL)
            except ProcessLookupError:
                pass


class DockerProvider:
    """Optional Linux worker image with a private directory mounted per run."""
    def __init__(self, image: str):
        self.image = image

    async def start(self, run, directory: Path, token: str, url: str, cleanup: float):
        environment = directory / 'worker.env'
        environment.write_text(f'LIVEKIT_URL={url}\nLIVEKIT_TOKEN={token}\nMUJOCO_GL=osmesa\n')
        environment.chmod(0o600)
        name = f"teleop-demo-{run['id']}"
        try:
            child = await asyncio.create_subprocess_exec('docker', 'run', '--detach', '--rm', '--init',
                '--name', name, '--label', f"teleop.demo.run={run['id']}", '--network', 'host',
                '--memory', '2g', '--cpus', '2', '--env-file', str(environment),
                '--mount', f'type=bind,source={directory},target=/run/session', self.image,
                'python', '-m', 'sims.demo.worker', '--id', run['id'], '--simulation', run['simulation_id'],
                '--directory', '/run/session', '--cleanup', str(cleanup), '--log',
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            _, error = await child.communicate()
            if child.returncode:
                raise RuntimeError(f'Docker could not start the simulation: {error.decode()[:300]}')
        finally:
            environment.unlink(missing_ok=True)
        return name

    def status(self, run):
        import subprocess
        try:
            result = subprocess.run(['docker', 'inspect', '--format',
                '{{.State.Running}} {{index .Config.Labels "teleop.demo.run"}}', run['pid']],
                capture_output=True, text=True, timeout=3)
            return result.returncode == 0 and result.stdout.strip() == f"true {run['id']}"
        except (KeyError, subprocess.SubprocessError):
            return False

    def stop(self, run):
        import subprocess
        if self.status(run):
            subprocess.run(['docker', 'kill', run['pid']], capture_output=True, timeout=5)
