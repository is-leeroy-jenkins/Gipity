FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8501
ENV APP_FILE=app.py
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

RUN chmod +x /app/startup.sh

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail "http://127.0.0.1:${PORT}/_stcore/health" || exit 1

CMD ["/app/startup.sh"]
