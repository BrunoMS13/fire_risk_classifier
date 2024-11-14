# Use a slim base image to reduce size
FROM python:3.11-slim

# Set a working directory
WORKDIR /app

# Install necessary packages for Poetry and remove after installation
RUN apt-get update && \
    apt-get install -y curl && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    apt-get remove -y curl && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Add Poetry's bin directory to PATH
ENV PATH="/root/.local/bin:$PATH"

# Enable Poetry's virtual environment in the project directory
ENV POETRY_VIRTUALENVS_IN_PROJECT=true

# Copy the project files
COPY . .

# Install dependencies without development packages
RUN poetry install --no-dev

# Set the command to run the application
CMD ["poetry", "run", "train"]
