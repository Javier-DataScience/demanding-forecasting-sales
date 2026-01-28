FROM python:3.11-slim

# Install system dependencies required by LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory and Python path
WORKDIR /app
ENV PYTHONPATH=/app

# Copy requirements first (better Docker cache usage)
COPY ../requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (src, data, models, etc.)
COPY . .
