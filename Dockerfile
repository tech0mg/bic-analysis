# ベースイメージ（Python 3.12 の slim）
FROM python:3.12-slim

# 作業ディレクトリ
WORKDIR /app

# 必要ファイルをコピーして依存をインストール
COPY requirements.txt .
RUN pip install --no‑cache‑dir -r requirements.txt

# ソースコードをすべてコピー
COPY . .

# Streamlit のポート
ENV STREAMLIT_SERVER_HEADLESS true
ENV STREAMLIT_SERVER_PORT 8501
EXPOSE 8501

# 起動コマンド
CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0"]