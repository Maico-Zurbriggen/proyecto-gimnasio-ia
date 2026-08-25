FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src src
RUN python -m pip install --no-cache-dir .

CMD ["python", "-c", "import gym_engine; print('Motor batch listo')"]
