FROM python:3.12-slim

WORKDIR /app

# CPU-only torch is much smaller than the CUDA build and is all serving needs.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
COPY configs/ ./configs/
COPY checkpoints/ ./checkpoints/

EXPOSE 8000
ENV MODEL_PATH=checkpoints/ckpt_step_8000.pt
ENV TOKENIZER_PATH=checkpoints/tokenizer.json
ENV CONFIG_PATH=configs/small.yaml

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
