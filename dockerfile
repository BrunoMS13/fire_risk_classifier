# Use a base image with Python 3.11
FROM python:3.11

# Set a working directory in the container
WORKDIR /app

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    export PATH="/root/.local/bin:$PATH" && \
    poetry --version

# Add Poetry's bin directory to PATH (for running `poetry` commands)
ENV PATH="/root/.local/bin:$PATH"

# Copy the project files into the container
COPY . /app

# Install project dependencies using Poetry
RUN poetry install

# Define the command to run the training script
CMD ["poetry", "run", "python", "fire_risk_classifier/train.py"]
