FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

COPY src/ ./src/

ENV OPENAI_API_KEY=""
ENV CHROMA_PERSIST_DIR="/data/chroma"

VOLUME ["/data/chroma", "/data/docs"]

ENTRYPOINT ["python", "-m", "ragpipe"]
CMD ["--help"]
