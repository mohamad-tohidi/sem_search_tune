FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y curl vim git && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

COPY . /workspace
WORKDIR /workspace
RUN uv run sync

CMD ["bash"]
