"""两个实验入口的参数解析：只验证"不再硬编码 Agent 名字"，不联网、不构造 Agent。

run_shop / probe 曾经把 --agent 写成 choices=("tuned","cold")，
接任何新 Agent 都会在 argparse 阶段就被拦住。这两条测试盯住这个回归。
"""

from __future__ import annotations

from scripts.probe import parse_args as probe_args
from scripts.run_shop import parse_args as shop_args


def test_run_shop_accepts_any_agent_name():
    args = shop_args(["--agent", "travel", "--split", "holdout", "--limit", "1"])
    assert args.agent == "travel"
    # 默认值保持 tuned，旧命令不受影响
    assert shop_args([]).agent == "tuned"


def test_probe_accepts_any_agent_name():
    args = probe_args(["--task", "M07", "--agent", "travel"])
    assert args.agent == "travel"
    assert probe_args(["--task", "M07"]).agent == "cold"
