FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
COPY --from=docker.io/astral/uv:latest /uv /uvx /bin/
WORKDIR /workspace

RUN apt-get update && apt-get install -y curl vim git && rm -rf /var/lib/apt/lists/*

COPY . /workspace
WORKDIR /workspace
RUN uv run sync

CMD ["bash"]
