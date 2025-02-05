# CUDA-enabled base image (Ubuntu 22.04 with CUDA 11.7, cuDNN 8)
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

# Set a working directory
WORKDIR /app

# Install essential packages and Poetry
RUN apt-get update && \
    apt-get install -y curl python3-dev build-essential && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Add Poetry's bin directory to PATH
ENV PATH="/root/.local/bin:$PATH"

# Enable Poetry's virtual environment in the project directory
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV PATH="/app/.venv/bin:$PATH"

# Copy project files
COPY . .

# Make sure your pyproject.toml (and poetry.lock if present) is included
# so that Poetry can install the correct dependencies including a
# GPU-compatible PyTorch wheel.

# Install Python dependencies
RUN poetry install

# Set the default command to run your training script
CMD ["poetry", "run", "train"]
