FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install sentence-transformers datasets accelerate

COPY . /workspace

CMD ["bash"]