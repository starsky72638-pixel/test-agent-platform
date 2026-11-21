#!/bin/sh
# 容器入口：先拉 mock_server，确认它活了，再把 api 顶到前台。
# api 必须是前台进程（exec），否则容器会立刻退出。
set -eu

echo "[entrypoint] starting mock_server on 127.0.0.1:8000"
python mock_server.py > /tmp/mock.log 2>&1 &
MOCK_PID=$!
trap 'kill "$MOCK_PID" 2>/dev/null || true' EXIT

# 等 mock 真的能应答再起 api：api 首次请求就会打 mock，
# 这里不等的话，容器刚起来那几秒的请求会一律 500。
i=0
while [ "$i" -lt 40 ]; do
  if python - <<'PY' 2>/dev/null
import sys
import httpx

try:
    r = httpx.get("http://127.0.0.1:8000/orders/1002", timeout=1.0)
    sys.exit(0 if r.status_code == 200 else 1)
except Exception:
    sys.exit(1)
PY
  then
    echo "[entrypoint] mock_server ready"
    break
  fi
  i=$((i + 1))
  sleep 0.5
done

if [ "$i" -ge 40 ]; then
  echo "[entrypoint] mock_server 未在 20s 内就绪，日志：" >&2
  cat /tmp/mock.log >&2 || true
fi

echo "[entrypoint] starting api on 0.0.0.0:8020"
exec uvicorn api.app:app --host 0.0.0.0 --port 8020 --workers 1 --no-access-log
