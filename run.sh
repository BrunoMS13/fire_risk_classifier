echo "Setting up Poetry and project dependencies..."
# Install Poetry if it's not installed already
if ! command -v poetry &> /dev/null
then
    echo "Poetry not found, installing..."
    curl -sSL https://install.python-poetry.org | python3 -
fi

# Set up the Python environment using Poetry
cd /tmp/test/fire_risk_classifier  # or wherever your project is located
poetry install  # This will install all the dependencies from pyproject.toml

# Run your script (e.g., training, testing, etc.)
echo "Running the training script..."
poetry run train  # Replace with your script name