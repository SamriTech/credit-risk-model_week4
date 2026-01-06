FROM python:3.13-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# This is the default, but docker-compose can override it
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
