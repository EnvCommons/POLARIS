FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt update && apt install -y \
    python3 \
    python3-pip \
    curl \
    git \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
RUN uv venv --python 3.11

# Copy application files
COPY requirements.txt /app/
COPY polaris.py /app/
COPY server.py /app/

# Install Python dependencies
RUN uv pip install -r /app/requirements.txt

# Expose server port
EXPOSE 8080

# Start server
CMD ["uv", "run", "python", "/app/server.py"]
