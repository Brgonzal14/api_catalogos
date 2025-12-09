FROM python:3.12-slim

WORKDIR /app

# Ya NO instalamos Java, porque para el prototipo solo usamos Excel/CSV.
# Si más adelante agregamos PDFs con Tabula/OCR, ahí lo vemos de nuevo.

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV PYTHONPATH=/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
