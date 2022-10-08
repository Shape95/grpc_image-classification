ARG BASE_IMAGE=ubuntu:22.04

# Compile image loosely based on pytorch compile image
FROM ${BASE_IMAGE} AS compile-image
ENV PYTHONUNBUFFERED TRUE

# Install Python and pip, and build-essentials if some requirements need to be compiled
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    python3-dev \
    python3-distutils \
    python3-venv \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && cd /tmp \
    && curl -O https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py

RUN python3 -m venv /home/venv

ENV PATH="/home/venv/bin:$PATH"

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN update-alternatives --install /usr/local/bin/pip pip /usr/local/bin/pip3 1

# The part above is cached by Docker for future builds
# We can now copy the requirements file from the local system
# and install the dependencies
COPY requirements.txt .

# Use the pre-build Python packages for PyTorch
RUN pip install --no-cache-dir -r requirements.txt

FROM pytorch/torchserve as production

# Copy dependencies after having built them
# COPY --from=compile-image /home/venv /home/venv

# We use curl for health checks on AWS Fargate
USER root
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# USER model-server

COPY deploy/config.properties /home/model-server/config.properties
COPY deploy/model-store/resnet_scripted_quantized.mar /home/model-server/model-store

RUN mkdir proto
COPY proto/classification_pb2.py /home/model-server/proto
COPY proto/classification_pb2_grpc.py /home/model-server/proto

COPY resnet.py /home/model-server
COPY server.py /home/model-server
RUN mkdir script
COPY script/resnet_scripted_quantized.pt /home/model-server/script

RUN pip install --upgrade pip
RUN pip install grpcio
RUN pip install grpcio-tools

CMD 'python server.py'