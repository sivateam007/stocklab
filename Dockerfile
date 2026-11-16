# Stocklab web app for Render (https://render.com)
# TA-Lib 0.7.1 ships manylinux wheels for cp312, so no C toolchain is needed.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements-web.txt .
RUN pip install -r requirements-web.txt

# Application code + the precomputed data it serves (data/ is ~92 MB).
COPY stocklab/ stocklab/
COPY app/ app/
COPY data/ data/
COPY .streamlit/ .streamlit/

# Render assigns $PORT (default 10000) and expects the app to bind to it.
EXPOSE 10000
CMD streamlit run app/dashboard.py --server.address=0.0.0.0 --server.port=${PORT:-10000} --server.headless=true --browser.gatherUsageStats=false
