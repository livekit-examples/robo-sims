# The reBot B601-DM emulated hardware alone, for drivers on other machines:
# MuJoCo behind the emulated Damiao bus, on /dev/ttyACM0 inside the container
# and on TCP port 5555. No ROS, no LiveKit, no GPU.
#
# Build from simulation/:   docker compose -f docker/rebot-bus/compose.yaml up --build -d

FROM python:3.12-slim-bookworm AS assets
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY sims /src/sims
ENV SIM_ASSETS_DIR=/assets
RUN cd /src && python -m sims fetch-assets rebot-arm

FROM python:3.12-slim-bookworm
ENV PIP_NO_CACHE_DIR=1 PIP_ROOT_USER_ACTION=ignore PYTHONUNBUFFERED=1
COPY requirements.txt requirements-dev.txt /opt/simulation/
# motorbridge is only for tests/rebot_motorbridge_client.py; livekit is unused here.
RUN pip install "$(grep '^mujoco' /opt/simulation/requirements.txt)" \
                "$(grep '^numpy' /opt/simulation/requirements.txt)" \
                "$(grep '^motorbridge' /opt/simulation/requirements-dev.txt)"
COPY --from=assets /assets /opt/simulation/assets
COPY sims /opt/simulation/sims
COPY tools /opt/simulation/tools
COPY tests /opt/simulation/tests
ENV PYTHONPATH=/opt/simulation SIM_ASSETS_DIR=/opt/simulation/assets
WORKDIR /opt/simulation
EXPOSE 5555
ENTRYPOINT ["python", "-m", "sims.rebot.bus", "--serial-link", "/dev/ttyACM0", \
            "--tcp-host", "0.0.0.0", "--tcp-port", "5555", "--status-file", "/tmp/rebot-sim-status.json"]
