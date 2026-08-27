FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
COPY data/ data/
RUN pip install --no-cache-dir -e ".[agent,dev]"
COPY tests/ tests/
EXPOSE 8000
CMD ["uvicorn", "oficio.service.api:app", "--host", "0.0.0.0", "--port", "8000"]
