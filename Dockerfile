FROM python:3.11-slim

WORKDIR /app

COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

COPY online_rpg/ online_rpg/
COPY run_server.py .

EXPOSE 8000

CMD ["python", "run_server.py", "--host", "0.0.0.0", "--port", "8000"]
