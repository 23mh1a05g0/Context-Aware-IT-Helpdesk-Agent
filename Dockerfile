FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY eval_5_turns.py .

# Create logs directory
RUN mkdir -p logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); assert r.status_code == 200" || exit 1

# Start FastAPI with uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
