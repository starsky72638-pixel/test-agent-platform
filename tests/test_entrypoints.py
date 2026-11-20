"""两个实验入口的守卫：不联网、不构造 Agent，只读参数与源码文本。

盯两类回归：
1. --agent 曾经写成 choices=("tuned","cold")，接新 Agent 在 argparse 阶段就被拦住；
2. scripts/ 曾经反向 import agents.shop 取服务地址，换后端就在探活阶段被挡住。
"""

from __future__ import annotations

from pathlib import Path

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


def test_shop_base_has_single_source():
    """服务地址只在 config 定义一次，shop 侧引用它而不是自己读环境变量。

    T5-a 之前 SHOP_BASE 是 shop_skill.py 自己 os.getenv 出来的，
    scripts/ 再反向 import 它——服务地址因此被绑死在某个具体 Agent 上。
    """
    root = Path(__file__).resolve().parent.parent
    source = (root / "agents" / "shop" / "tools" / "shop_skill.py").read_text(
        encoding="utf-8"
    )

    assert "from config.settings import SERVICE_BASE" in source
    assert "SHOP_BASE = SERVICE_BASE" in source
    assert 'os.getenv("SHOP_BASE_URL")' not in source, "shop 不该再自己读服务地址"


def test_entrypoints_do_not_import_concrete_agent():
    """两个实验入口不许 import 任何具体 Agent 的模块。

    README 第 4 节承诺 `run_shop.py --agent <name>` 能跑任意新 Agent。
    只要 scripts/ 里还留着 `from agents.shop import ...`，
    换一个不连 shop 后端的 Agent 就会在探活阶段被挡住。
    """
    root = Path(__file__).resolve().parent.parent
    for name in ("run_shop.py", "probe.py"):
        source = (root / "scripts" / name).read_text(encoding="utf-8")
        assert "from agents.shop" not in source, f"{name} 仍依赖具体 Agent 的模块"


def test_shop_output_filename_is_windows_safe():
    """`--agent wf:canvas-demo` 会拼出含冒号的文件名，Windows 上 open() 直接 OSError。

    更糟的是它崩在"评测跑完、数字已打印"之后：数字有了、轨迹没落盘，
    事后无法复现。这条钉住 sanitize。
    """
    from scripts.run_shop import safe_filename

    assert safe_filename("wf:canvas-demo") == "wf-canvas-demo"
    assert ":" not in safe_filename("a:b/c\\d")
    assert safe_filename("") == "agent"
