FROM python:3.11-slim

WORKDIR /app

# 系統依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python 依賴（先 copy requirements 利用 Docker cache）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 預先下載 embedding 模型到 image 裡（避免每次啟動都下載）
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-zh-v1.5')"

# 下載 Spacy 中文 NLP 模型供 Presidio 使用
RUN python -m spacy download zh_core_web_sm

# 專案程式碼由 docker-compose volume mount 進來
# 所以不 COPY 程式碼，保持開發時即時同步

CMD ["python", "-m", "src.pipeline"]
