FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src src
RUN python -m pip install --no-cache-dir .

CMD ["python", "-m", "uvicorn", "gym_engine.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
