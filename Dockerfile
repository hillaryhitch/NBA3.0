FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY optimizer_api.py .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "optimizer_api:app", "--host", "0.0.0.0", "--port", "8000"]