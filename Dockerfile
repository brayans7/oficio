FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
COPY data/ data/
RUN pip install --no-cache-dir -e ".[agent,dev]"
COPY tests/ tests/
EXPOSE 8000
CMD ["python", "-c", "print('Oficio v0.1 — API arrives in T12. Run: pytest')"]
