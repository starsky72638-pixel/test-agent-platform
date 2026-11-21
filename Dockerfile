FROM python:3.12-slim

# 时区跟着部署地走，日志里的时间才不会和宿主机差 8 小时
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

WORKDIR /app

# 先装依赖：代码改动时这一层可以命中缓存。
# 用 requirements.serve.txt（钉死版本）而不是 requirements.txt：
# 后者是开发环境用的宽松约束，pip 会回溯上百个版本，在国内网络下经常超时后
# 报出假的 "ResolutionImpossible"。
COPY requirements.serve.txt ./
RUN pip install --no-cache-dir \
    -i https://mirrors.cloud.tencent.com/pypi/simple \
    --trusted-host mirrors.cloud.tencent.com \
    --timeout 120 --retries 5 \
    -r requirements.serve.txt

# 用 uid 1000 运行：宿主机 ubuntu 也是 1000，
# 挂载卷里新建的文件属主一致，VSCode 里还能改；
# 否则容器以 root 写入，宿主上变成 root 所有，普通用户改不动。
RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/data /app/runs

COPY . .
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8020

# 一个容器跑两个进程（mock_server + uvicorn）：
# mock_server 写死监听 127.0.0.1:8000，只有同容器才能访问，
# 拆成两个容器就必须改源码，不划算。
CMD ["sh", "docker/entrypoint.sh"]
