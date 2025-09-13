# Dockerfile
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl vim ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy benchmark code
COPY benchmark_entry.py /app/benchmark_entry.py

# Default command
ENTRYPOINT ["python", "benchmark_entry.py"]
