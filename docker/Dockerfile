# Use Python 3.12 slim as the base image
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies required by pydub (ffmpeg)
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file from the project root
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files from the context (project root)
COPY . .

# Expose the internal port for Streamlit (8501)
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
