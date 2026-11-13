"""用例生成层：把 OpenAPI 文档压成契约文本，让模型输出结构化用例。"""

from __future__ import annotations

import json
import re

from langchain_core.language_models import BaseChatModel

SYSTEM = "你是接口测试用例生成器。只输出 JSON 数组，不要解释，不要 Markdown 代码块。"

PROMPT = """下面是接口文档的精简契约：

{contract}

请为每个接口生成测试用例，规则：
1. 每个接口至少 3 条：positive（正向）、boundary（边界/越界）、negative（类型错误/缺参）；
2. 只能使用文档里真实存在的 path、method、参数名，不要发明字段；
3. 期望状态码按文档语义判断：参数越界或类型错误=422，资源不存在=404，创建成功=201，查询成功=200；
4. 路径参数（如 {{id}}）放进 path_params，不要去拼字符串；
5. 严格输出 JSON 数组，元素结构如下：

{{"path": "/todos", "method": "GET", "query": {{"limit": 0}}, "path_params": {{}}, "body": null, "category": "boundary", "expect_status": 422, "title": "limit 越界下限"}}

只输出数组本身。"""

_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}


def _resolve(spec: dict, node: dict) -> dict:
    """展开 $ref，非引用原样返回。"""
    if not isinstance(node, dict) or "$ref" not in node:
        return node or {}
    ref = node["$ref"]
    if not ref.startswith("#/"):
        return {}
    cur: object = spec
    for part in ref[2:].split("/"):
        if not isinstance(cur, dict):
            return {}
        cur = cur.get(part, {})
    return cur if isinstance(cur, dict) else {}


def _range(schema: dict) -> str:
    """把 min/max 拼成 (integer,1..100)，让模型看到边界。"""
    lo, hi = schema.get("minimum"), schema.get("maximum")
    if lo is None and hi is None:
        return ""
    return f",{lo if lo is not None else '-'}..{hi if hi is not None else '-'}"


def summarize_contract(spec: dict) -> str:
    """把 OpenAPI 压成每行一个接口的文本，比塞整份 JSON 省 token 也更准。"""
    lines: list[str] = []
    for path, methods in spec.get("paths", {}).items():
        for method, op in methods.items():
            if method.upper() not in _METHODS:
                continue
            params = []
            for raw in op.get("parameters", []):
                p = _resolve(spec, raw)
                sch = _resolve(spec, p.get("schema", {}))
                params.append(f'{p.get("name")}({sch.get("type", "?")}{_range(sch)})')

            body = ""
            schema = (
                op.get("requestBody", {})
                .get("content", {})
                .get("application/json", {})
                .get("schema")
            )
            if schema:
                schema = _resolve(spec, schema)
                props = ", ".join(
                    f'{k}:{_resolve(spec, v).get("type", "?")}'
                    for k, v in (schema.get("properties") or {}).items()
                )
                body = f'  body{{{props}}} required={schema.get("required", [])}'

            lines.append(
                f'{method.upper()} {path}  params[{", ".join(params) or "-"}]{body}'
                f'  // {op.get("summary", "")}'
            )
    return "\n".join(lines)


def _extract_json_array(text: str) -> list:
    """模型偶尔会加代码围栏或前后废话，这里做三层兜底。"""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t).strip()
    try:
        data = json.loads(t)
    except json.JSONDecodeError:
        i, j = t.find("["), t.rfind("]")
        if i == -1 or j <= i:
            raise ValueError(f"模型没有返回 JSON 数组，原文前 300 字：\n{text[:300]}") from None
        data = json.loads(t[i : j + 1])

    if isinstance(data, dict):
        for key in ("cases", "test_cases", "data", "results"):
            if isinstance(data.get(key), list):
                return data[key]
        raise ValueError("返回的是对象，但找不到用例数组")
    if not isinstance(data, list):
        raise ValueError(f"返回的既不是数组也不是对象：{type(data).__name__}")
    return data


def normalize_cases(raw: list) -> tuple[list[dict], int]:
    """补齐字段、丢弃非法项。丢了多少条要打印出来，这是生成质量的一部分。"""
    cases: list[dict] = []
    dropped = 0
    for i, c in enumerate(raw, 1):
        if not isinstance(c, dict) or not c.get("path") or not c.get("method"):
            dropped += 1
            continue
        try:
            expect = int(c.get("expect_status"))
        except (TypeError, ValueError):
            dropped += 1
            continue
        missing = [
            key for key in re.findall(r"{(\w+)}", str(c["path"]))
            if key not in (c.get("path_params") or {})
        ]
        if missing:
            dropped += 1
            continue
        cases.append(
            {
                "id": f"C{i:02d}",
                "title": str(c.get("title") or f'{c["method"]} {c["path"]}')[:60],
                "path": str(c["path"]),
                "method": str(c["method"]).upper(),
                "query": c.get("query") or {},
                "path_params": c.get("path_params") or {},
                "body": c.get("body"),
                "category": str(c.get("category") or "positive"),
                "expect_status": expect,
            }
        )
    return cases, dropped


def generate_cases(llm: BaseChatModel, spec: dict, extra_rules: str = "") -> tuple[list[dict], int]:
    """一次调用生成全套用例。返回值：(用例列表, 被丢弃条数)。"""
    prompt = PROMPT.format(contract=summarize_contract(spec))
    if extra_rules:
        prompt = f"{prompt}\n\n额外规则：\n{extra_rules}"

    msg = llm.invoke([("system", SYSTEM), ("human", prompt)])
    text = msg.content if hasattr(msg, "content") else str(msg)
    if isinstance(text, list):  # 部分模型返回分片列表
        text = "".join(str(x) for x in text)

    cases, dropped = normalize_cases(_extract_json_array(text))
    if not cases:
        raise ValueError("所有用例都被丢弃，说明模型输出格式完全不对")
    return cases, dropped
