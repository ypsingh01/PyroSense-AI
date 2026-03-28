FROM python:3.10-slim AS builder

WORKDIR /app
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt requirements-dev.txt ./
RUN pip install --upgrade pip && pip install --prefix=/install -r requirements.txt

FROM python:3.10-slim AS runtime

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY --from=builder /install /usr/local
COPY . /app

EXPOSE 8501 8000

ENTRYPOINT ["python", "-m", "supervisor", "-c", "/app/supervisord.conf"]

