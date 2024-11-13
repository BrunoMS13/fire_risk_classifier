# Use an NVIDIA GPU-compatible base image
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Install dependencies for Python 3.11
RUN apt-get update && apt-get install -y \
    curl \
    git \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y python3.11 python3.11-venv python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as the default Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Verify Python version
RUN python3 --version

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3.11 -

# Add Poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

# Set the working directory in the container
WORKDIR /app

# Copy Poetry files and install dependencies
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root

# Copy the project files into the container
COPY . .

# Default command to train the model
CMD ["poetry", "run", "python", "train.py"]