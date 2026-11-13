"""执行层：逐条跑用例，用代码比对状态码给出通过与否。"""

from __future__ import annotations

import time

import httpx


def _build_url(base_url: str, path: str, path_params: dict) -> str:
    for key, value in (path_params or {}).items():
        path = path.replace("{" + str(key) + "}", str(value))
    return base_url.rstrip("/") + path


def _verdict(case: dict, actual: int | None) -> str:
    """规则化初判，D3 会在这里接入 LLM 归因。"""
    if actual is None:
        return "环境问题"
    expect = case["expect_status"]
    if actual >= 500:
        return "疑似产品缺陷"
    if expect in (400, 422) and actual < 300:
        return "疑似产品缺陷"
    if expect == 404 and actual == 200:
        return "疑似产品缺陷"
    if expect == 201 and actual == 200:
        return "疑似产品缺陷"
    return "待人工确认"


def run_case(client: httpx.Client, base_url: str, case: dict) -> dict:
    url = _build_url(base_url, case["path"], case["path_params"])
    started = time.perf_counter()
    try:
        resp = client.request(
            case["method"],
            url,
            params=case["query"] or None,
            json=case["body"],
        )
    except Exception as exc:
        return {
            **case,
            "actual_status": None,
            "passed": False,
            "verdict": "环境问题",
            "detail": f"{type(exc).__name__}: {exc}",
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "response": "",
        }

    elapsed = int((time.perf_counter() - started) * 1000)
    passed = resp.status_code == case["expect_status"]
    return {
        **case,
        "actual_status": resp.status_code,
        "passed": passed,
        "verdict": "通过" if passed else _verdict(case, resp.status_code),
        "detail": "" if passed else f'期望 {case["expect_status"]}，实际 {resp.status_code}',
        "elapsed_ms": elapsed,
        "response": resp.text[:220],
    }


def run_cases(cases: list[dict], base_url: str, timeout: float = 5.0) -> list[dict]:
    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        return [run_case(client, base_url, c) for c in cases]


def summarize(results: list[dict]) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    by_category: dict[str, dict[str, int]] = {}
    for r in results:
        bucket = by_category.setdefault(r["category"], {"total": 0, "passed": 0})
        bucket["total"] += 1
        if r["passed"]:
            bucket["passed"] += 1
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total * 100, 1) if total else 0.0,
        "by_category": by_category,
        "suspect_defect": sum(1 for r in results if r["verdict"] == "疑似产品缺陷"),
        "elapsed_ms": sum(r["elapsed_ms"] for r in results),
    }
