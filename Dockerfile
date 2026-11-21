FROM python:3.12-slim

# 时区跟着部署地走，日志里的时间才不会和宿主机差 8 小时
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

WORKDIR /app

# 先装依赖：代码改动时这一层可以命中缓存
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8020

# 一个容器跑两个进程（mock_server + uvicorn）：
# mock_server 写死监听 127.0.0.1:8000，只有同容器才能访问，
# 拆成两个容器就必须改源码，不划算。
CMD ["sh", "docker/entrypoint.sh"]
