"""SFT 训练脚本（LoRA）。本机显存 6G，只建议跑 0.5B~1.5B 的小模型做配方验证。

数据来自 tests/feedback.py 生成的 data/generated/train_sft.jsonl：
    {"messages": [system, user, assistant(+tool_calls), tool, assistant], "meta": {...}}

为什么不用 trl 的 SFTTrainer：
  trl 不同版本对 chat template、packing、loss mask 的参数名一直在改，
  这份脚本的目标是"配方可读、能在小模型上跑通"，所以只依赖 transformers + peft。

依赖（当前环境未安装，本脚本未经运行验证）：
    pip install torch transformers peft datasets accelerate

已知简化（写在这里，避免当成能直接上生产的东西）：
  1. loss 目前覆盖整条序列，包含 system 与 tool 返回。
     真正训练应该只对 assistant 段算 loss，否则模型会去背工具返回的 JSON。
  2. 没有做长度截断后的样本过滤，超长样本会被 tokenizer 截断。
  3. 没有 packing，短样本会浪费算力。

用法：
    python train/sft_lora.py --data data/generated/train_sft.jsonl --model Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="feedback.py 生成的 train_sft.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="train/out")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=2048)
    args = ap.parse_args()

    # 依赖放在函数里导入：没装依赖时 `python -c "import"` 不会直接炸，
    # 而且这份脚本的主要用途是"读配方"，不是被 import。
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    data_path = Path(args.data)
    rows = load_jsonl(data_path)
    if not rows:
        raise SystemExit(f"{data_path} 是空的，先跑 tests/feedback.py")
    print(f"样本 {len(rows)} 条，来自 {data_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def to_text(row: dict) -> str:
        """把 messages 套用模型自带的 chat template，得到一条训练文本。

        工具调用已经写在 assistant 的 tool_calls 里，部分 template 不支持渲染，
        这种情况下退化成"只拼 content"，并在样本里留下痕迹便于排查。
        """
        messages = row["messages"]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            parts = []
            for m in messages:
                parts.append(f"[{m.get('role')}] {m.get('content') or ''}")
            return "\n".join(parts)

    texts = [to_text(r) for r in rows]

    def tokenize(batch: dict) -> dict:
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_len,
            padding=False,
        )
        # 因果语言模型：labels 就是 input_ids 本身
        enc["labels"] = list(enc["input_ids"])
        return enc

    ds = Dataset.from_dict({"text": texts}).map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            # 只挂在注意力与 MLP 的投影层上，6G 显存下这是能跑的最小集合
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.bs,
            gradient_accumulation_steps=args.accum,
            learning_rate=args.lr,
            logging_steps=1,
            save_strategy="epoch",
            fp16=True,
            report_to=[],
        ),
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"已保存到 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
