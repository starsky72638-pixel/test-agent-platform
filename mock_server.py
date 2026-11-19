"""被测环境：一个电商后端，供被评 Agent 调用。

五个接口对应 Agent 的五个 skill：
  售前  GET /products/search    GET /products/{id}
  售后  GET /orders/{id}        GET /logistics/{id}     POST /refund
  评测  POST /admin/reset       清空退款记录，保证每次回放初始状态一致

两处刻意设计（不是 bug，是考点）：
  A. 状态枚举不统一：/orders 返回中文（"已发货"），/logistics 返回英文（"shipped"）
     → 评 Agent 的跨工具对齐能力
  B. 订单 1002 已发货但 eta 为 null、1003 未发货且无轨迹
     → 评 Agent 会不会编造"预计明天送达"（幻觉）

运行：
    python mock_server.py
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

# ---------------------------------------------------------------- 数据

PRODUCTS: dict[int, dict] = {
    1: {
        "id": 1,
        "title": "无线降噪耳机 Pro",
        "price": 899.0,
        "stock": 42,
        "rating": 4.7,
        "specs": {"降噪": "主动降噪 45dB", "续航": "30 小时", "连接": "蓝牙 5.3"},
        "suitable_for": "通勤、办公、长途飞行",
        "desc": "双麦克风通话降噪，支持多设备切换，附赠三种尺寸耳塞。",
    },
    2: {
        "id": 2,
        "title": "机械键盘 K8",
        "price": 429.0,
        "stock": 15,
        "rating": 4.5,
        "specs": {"轴体": "红轴", "配列": "87 键", "连接": "有线 / 蓝牙 / 2.4G"},
        "suitable_for": "编程、文字工作",
        "desc": "PBT 键帽，支持热插拔，三模连接。",
    },
    3: {
        "id": 3,
        "title": "便携充电宝 20000mAh",
        "price": 159.0,
        "stock": 88,
        "rating": 4.3,
        "specs": {"容量": "20000mAh", "输出": "22.5W", "接口": "USB-C ×1 / USB-A ×2"},
        "suitable_for": "出差、露营",
        "desc": "支持双向快充，可上飞机。",
    },
    4: {
        "id": 4,
        "title": "智能手表 S2",
        "price": 1299.0,
        "stock": 0,
        "rating": 4.6,
        "specs": {"屏幕": "1.43 英寸 AMOLED", "续航": "14 天", "防水": "5ATM"},
        "suitable_for": "运动、健康监测",
        "desc": "支持血氧与心率监测，兼容 iOS / Android。",
    },
    5: {
        "id": 5,
        "title": "27 寸 4K 显示器",
        "price": 2199.0,
        "stock": 7,
        "rating": 4.8,
        "specs": {"分辨率": "3840×2160", "刷新率": "60Hz", "接口": "HDMI2.1 ×2 / DP1.4"},
        "suitable_for": "设计、剪辑、多窗口办公",
        "desc": "出厂校色 ΔE<2，支持 65W 反向供电。",
    },
    6: {
        "id": 6,
        "title": "蓝牙音箱 Mini",
        "price": 299.0,
        "stock": 120,
        "rating": 4.2,
        "specs": {"功率": "10W", "续航": "12 小时", "防水": "IPX7"},
        "suitable_for": "户外、浴室",
        "desc": "支持双机互联立体声。",
    },
}

ORDERS: dict[int, dict] = {
    1001: {
        "order_id": 1001,
        "status": "已签收",
        "product_id": 1,
        "amount": 899.0,
        "created_at": "2026-11-02",
    },
    1002: {
        "order_id": 1002,
        "status": "已发货",
        "product_id": 2,
        "amount": 429.0,
        "created_at": "2026-11-08",
    },
    1003: {
        "order_id": 1003,
        "status": "待支付",
        "product_id": 6,
        "amount": 299.0,
        "created_at": "2026-11-11",
    },
}

# 考点 B：1002 的 eta 为空，1003 还没发货
LOGISTICS: dict[int, dict] = {
    1001: {
        "order_id": 1001,
        "carrier": "顺丰速运",
        "tracking_no": "SF1234567890",
        "status": "delivered",
        "eta": "2026-11-05",
        "traces": [
            {"time": "2026-11-04 21:10", "desc": "已到达武汉转运中心"},
            {"time": "2026-11-05 09:32", "desc": "已签收，签收人：本人"},
        ],
    },
    1002: {
        "order_id": 1002,
        "carrier": "中通快递",
        "tracking_no": "ZT9876543210",
        "status": "shipped",
        "eta": None,
        "traces": [
            {"time": "2026-11-09 14:05", "desc": "已揽收"},
            {"time": "2026-11-10 08:41", "desc": "已发出，下一站武汉分拨中心"},
        ],
    },
    1003: {
        "order_id": 1003,
        "carrier": None,
        "tracking_no": None,
        "status": "pending",
        "eta": None,
        "traces": [],
    },
}

REFUNDS: dict[int, dict] = {}

# ---------------------------------------------------------------- 故障注入
# 只为一件事存在：验证"工具失败之后 Agent 有没有自救"。
# 不注入时（默认）行为与以前完全一致。
#
# 用环境变量开启，因为它必须能被同一份代码反复对照：
#   FAULT_PATH=/products/search FAULT_STATUS=500 FAULT_TIMES=1 python mock_server.py
#
# reset 时把剩余次数拨回初始值，所以 probe 的每组都从"会失败一次"开始，
# 两组的初始条件严格相同。
FAULT_PATH = (os.getenv("FAULT_PATH") or "").strip()
FAULT_STATUS = int(os.getenv("FAULT_STATUS") or "500")
FAULT_TIMES = int(os.getenv("FAULT_TIMES") or "1")

FAULTS: dict[str, int] = {}


def reset_faults() -> None:
    FAULTS.clear()
    if FAULT_PATH:
        FAULTS[FAULT_PATH] = FAULT_TIMES


reset_faults()


# ---------------------------------------------------------------- 服务


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # 关掉默认访问日志，保持控制台干净
        pass

    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _consume_fault(self, path: str) -> bool:
        """命中注入的故障就回错误并扣一次；返回 True 表示已经应答。"""
        for prefix, left in list(FAULTS.items()):
            if left > 0 and path.startswith(prefix):
                FAULTS[prefix] = left - 1
                # 必须打出来：不打印的话，"故障到底有没有真的打进去"只能靠
                # 工具调用次数反推，那种推断出错时会把结论带偏。
                print(f"[fault] {path} -> {FAULT_STATUS}（本条还剩 {left - 1} 次）", flush=True)
                self._send(FAULT_STATUS, {"error": "injected fault", "path": path})
                return True
        return False

    # ------------------------------------------------------------ GET

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path, query = parsed.path, parse_qs(parsed.query)

        if self._consume_fault(path):
            return

        if path == "/products/search":
            return self._search_products(query)
        if path.startswith("/products/"):
            return self._get_product(path.split("/")[-1])
        if path.startswith("/orders/"):
            return self._get_order(path.split("/")[-1])
        if path.startswith("/logistics/"):
            return self._get_logistics(path.split("/")[-1])

        return self._send(404, {"error": "no route"})

    def _search_products(self, query: dict) -> None:
        keyword = (query.get("q") or [""])[0].strip().lower()
        items = list(PRODUCTS.values())
        if keyword:
            items = [
                p for p in items
                if keyword in p["title"].lower()
                or keyword in p["suitable_for"].lower()
                or keyword in p["desc"].lower()
            ]

        for key, op in (("min_price", "min"), ("max_price", "max")):
            raw = (query.get(key) or [""])[0]
            if raw:
                try:
                    bound = float(raw)
                except ValueError:
                    return self._send(422, {"error": f"{key} must be a number"})
                items = [p for p in items if p["price"] >= bound] if op == "min" \
                    else [p for p in items if p["price"] <= bound]

        brief = [
            {
                "id": p["id"],
                "title": p["title"],
                "price": p["price"],
                "stock": p["stock"],
                "rating": p["rating"],
            }
            for p in sorted(items, key=lambda x: -x["rating"])
        ]
        return self._send(200, {"total": len(brief), "items": brief})

    def _get_product(self, raw_id: str) -> None:
        try:
            pid = int(raw_id)
        except ValueError:
            return self._send(422, {"error": "product id must be an integer"})
        product = PRODUCTS.get(pid)
        if product is None:
            return self._send(404, {"error": "product not found"})
        return self._send(200, product)

    def _get_order(self, raw_id: str) -> None:
        try:
            oid = int(raw_id)
        except ValueError:
            return self._send(422, {"error": "order id must be an integer"})
        order = ORDERS.get(oid)
        if order is None:
            return self._send(404, {"error": "order not found"})
        return self._send(200, order)

    def _get_logistics(self, raw_id: str) -> None:
        try:
            oid = int(raw_id)
        except ValueError:
            return self._send(422, {"error": "order id must be an integer"})
        if oid not in ORDERS:
            return self._send(404, {"error": "order not found"})
        return self._send(200, LOGISTICS[oid])

    # ------------------------------------------------------------ POST

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return self._send(400, {"error": "invalid json"})
        if parsed.path == "/admin/reset":
            REFUNDS.clear()
            reset_faults()
            return self._send(200, {"ok": True})
        if self._consume_fault(parsed.path):
            return
        if parsed.path == "/refund":
            return self._refund(payload)

        return self._send(404, {"error": "no route"})

    def _refund(self, payload: dict) -> None:
        raw_order = payload.get("order_id")
        if raw_order is None:
            return self._send(422, {"error": "order_id is required"})
        try:
            oid = int(raw_order)
        except (TypeError, ValueError):
            return self._send(422, {"error": "order_id must be an integer"})

        # 参数校验放在业务校验之前：
        # 现在缺 reason 会先撞上"已申请过退款"的 409，把参数错误报成了业务冲突。
        if not str(payload.get("reason") or "").strip():
            return self._send(422, {"error": "reason is required"})

        order = ORDERS.get(oid)
        if order is None:
            return self._send(404, {"error": "order not found"})
        if order["status"] == "待支付":
            return self._send(422, {"error": "订单未支付，无法退款"})
        if oid in REFUNDS:
            return self._send(409, {"error": "该订单已申请过退款"})

        refund = {
            "refund_id": f"RF{oid}{len(REFUNDS) + 1:02d}",
            "order_id": oid,
            "status": "审核中",
            "amount": order["amount"],
            "eta_days": 3,
        }
        REFUNDS[oid] = refund
        return self._send(200, refund)


def make_server(port: int = 8000) -> HTTPServer:
    return HTTPServer(("127.0.0.1", port), Handler)


if __name__ == "__main__":
    server = make_server()
    print("被测环境已启动：http://127.0.0.1:8000")
    print("  GET  /products/search  /products/{id}  /orders/{id}  /logistics/{id}")
    print("  POST /refund   /admin/reset")
    print("Ctrl+C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
