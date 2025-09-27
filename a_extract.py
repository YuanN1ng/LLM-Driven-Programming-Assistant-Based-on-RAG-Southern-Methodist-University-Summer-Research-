#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
import json
from typing import List       # ← 兼容 3.7/3.8

# ---------- 配置区 ----------
INPUT_JSONL_PATH = r"data\test.jsonl"
OUTPUT_TXT_PATH  = r"data\codes_top50.txt"
MAX_RECORDS      = 50
SEPARATOR        = "\n# === END_OF_CODE ===\n"
# ----------------------------

def extract_codes(jsonl_path: Path, limit: int) -> List[str]:
    codes: List[str] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if len(codes) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] 第 {line_num} 行解析失败：{e}")
                continue
            code = obj.get("code")
            if code is not None:
                codes.append(code)
            else:
                print(f"[warn] 第 {line_num} 行缺少 'code' 字段")
    return codes

def main():
    in_path = Path(INPUT_JSONL_PATH).expanduser().resolve()
    if not in_path.is_file():
        raise FileNotFoundError(f"找不到文件：{in_path}")

    codes = extract_codes(in_path, MAX_RECORDS)
    print(f"[info] 共提取到 {len(codes)} 段代码（上限 {MAX_RECORDS}）")
    if len(codes) < MAX_RECORDS:
        print("[info] 文件记录不足 MAX_RECORDS，上限未触达")

    if OUTPUT_TXT_PATH:
        out_path = Path(OUTPUT_TXT_PATH).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)        # ← 新增
        out_path.write_text(SEPARATOR.join(codes) + "\n", encoding="utf-8")
        print(f"[info] 已写入 → {out_path}")
    else:
        print("\n\n".join(codes))

if __name__ == "__main__":
    main()
