FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System packages needed for numpy wheels and general builds
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md uv.lock* ./
COPY .env .env
COPY src ./src
COPY main.py ./main.py

# Install the application and runtime dependencies
RUN python -m pip install --upgrade pip \
    && python -m pip install "uvicorn[standard]" \
    && python -m pip install .

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
