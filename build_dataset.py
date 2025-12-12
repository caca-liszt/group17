#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build retrieval pairs (positives/negatives) using precomputed embeddings.

Inputs (match your files):
- jobs_csv:    must have columns [JD_ID, description]
- resumes_csv: must have columns [ID_variant, full_name, text_full_injected, group]
- job_embeds_pkl / resume_embeds_pkl: embeddings for jobs/resumes (same model + same dim)

Output:
- CSV with columns:
  id, resume_text, resume_text_withname, job_desc, name, group, label
  (optionally: sim_score, rank if --add_scores)

Example:
python build_dataset_from_embeddings.py \
  --jobs_csv /mnt/data/jds_filtered_isco2_conf060.csv \
  --resumes_csv /mnt/data/resumes_full_injected.csv \
  --job_embeds_pkl /mnt/data/description_embeddings.pkl \
  --resume_embeds_pkl /mnt/data/resume_embeddings_full.pkl \
  --topk 5 --neg_per_pos 1 --add_scores \
  --output_csv /mnt/data/retrieval_pairs_with_names_from_embeddings.csv
"""

import argparse
import csv
import io
import pickle
import re
from typing import Tuple, Optional, List, Any

import numpy as np
import pandas as pd


# ---------- PKL 兼容读取（修复 numpy._core 反序列化） ----------

class _NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # 某些环境把 numpy._core.* 序列化进去，老环境需映射到 numpy.core
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def _load_pickle_compat(path: str) -> Any:
    with open(path, "rb") as f:
        data = f.read()
    return _NumpyCompatUnpickler(io.BytesIO(data)).load()


def load_embeds_pkl(path: str) -> Tuple[Optional[List[str]], np.ndarray]:
    """
    兼容常见结构，返回 (ids, embeds)
      - dict: {"embeddings": array/list, "ids": [...]}（ids 可无）
      - list[dict]: [{"id": "...", "embedding": ...}, ...]
      - ndarray / list-of-lists: 按顺序返回，无 ids
    """
    obj = _load_pickle_compat(path)

    # case 1: dict with "embeddings" (and maybe "ids")
    if isinstance(obj, dict) and "embeddings" in obj:
        embs = np.asarray(obj["embeddings"], dtype=np.float32)
        ids = obj.get("ids", None)
        return ids, embs

    # case 2: list of dicts with "id" and "embedding"
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        ids, embs = [], []
        for item in obj:
            ids.append(item.get("id"))
            embs.append(np.asarray(item.get("embedding"), dtype=np.float32))
        return ids, np.vstack(embs).astype(np.float32)

    # case 3: raw ndarray (or list-of-lists)
    if isinstance(obj, np.ndarray):
        return None, obj.astype(np.float32)
    if isinstance(obj, list):
        arr = np.asarray(obj, dtype=np.float32)
        return None, arr

    raise ValueError(f"Unrecognized pickle structure: {type(obj)} from {path}")


# ---------- 文本清洗与工具 ----------

def clean_job_desc(s: Any) -> str:
    """JD `description` 看起来像 ['...'] 字符串，这里把外壳与引号去掉。"""
    if pd.isna(s):
        return ""
    s = str(s)
    s = re.sub(r"^\s*\[", "", s)
    s = re.sub(r"\]\s*$", "", s)
    s = s.replace("'", " ").replace('"', " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def strip_leading_name(text: Any, full_name: Any) -> str:
    """把注入文本首行的全名/Name: 前缀去掉，得到无姓名正文。"""
    text = "" if pd.isna(text) else str(text)
    full_name = "" if pd.isna(full_name) else str(full_name)
    lines = text.splitlines()
    if lines and lines[0].strip().lower().startswith(full_name.strip().lower()):
        return "\n".join(lines[1:]).lstrip()
    if lines and lines[0].strip().lower().startswith("name:"):
        return "\n".join(lines[1:]).lstrip()
    return text


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / norms


# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs_csv", required=True)
    ap.add_argument("--resumes_csv", required=True)
    ap.add_argument("--job_embeds_pkl", required=True)
    ap.add_argument("--resume_embeds_pkl", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--neg_per_pos", type=int, default=1)
    ap.add_argument("--add_scores", action="store_true", help="输出 sim_score 与 rank")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_csv", required=True)
    args = ap.parse_args()

    # 读取 CSV
    jobs = pd.read_csv(args.jobs_csv)
    resumes = pd.read_csv(args.resumes_csv)

    # 加载嵌入
    j_ids, J = load_embeds_pkl(args.job_embeds_pkl)      # [Nj, D]
    r_ids, R = load_embeds_pkl(args.resume_embeds_pkl)   # [Nr, D]

    # 基本一致性校验
    if J.shape[0] != len(jobs):
        raise ValueError(f"JD embeddings rows ({J.shape[0]}) != jobs CSV rows ({len(jobs)})")
    if R.shape[0] != len(resumes):
        raise ValueError(f"Resume embeddings rows ({R.shape[0]}) != resumes CSV rows ({len(resumes)})")
    if J.shape[1] != R.shape[1]:
        raise ValueError(f"Embedding dims differ: {J.shape[1]} vs {R.shape[1]}")

    # 归一化与相似度
    Jn = l2_normalize(J)
    Rn = l2_normalize(R)
    sims = Jn @ Rn.T  # [Nj, Nr]

    # 准备文本列
    jobs["job_desc"] = jobs["description"].apply(clean_job_desc)
    resumes["resume_text_withname"] = resumes["text_full_injected"].astype(str)
    resumes["name"] = resumes["full_name"].astype(str)
    resumes["resume_text"] = [
        strip_leading_name(t, n) for t, n in zip(resumes["resume_text_withname"], resumes["name"])
    ]
    resumes["group"] = resumes["group"].astype(str)

    # 预提取列
    res_id = resumes["ID_variant"].tolist()
    res_group = resumes["group"].tolist()
    res_name = resumes["name"].tolist()
    res_withname = resumes["resume_text_withname"].tolist()
    res_plain = resumes["resume_text"].tolist()

    job_id = jobs["JD_ID"].tolist()
    job_text = jobs["job_desc"].tolist()

    rng = np.random.RandomState(args.seed)
    rows = []

    # 为每个 JD 做 top-k 命中 + 等量负样本
    for j_idx in range(J.shape[0]):
        sim_row = sims[j_idx]
        order = np.argsort(-sim_row)
        pos_idx = order[:args.topk]
        neg_pool = order[args.topk:]

        # positives
        for rank_pos, r_idx in enumerate(pos_idx, start=1):
            rid = res_id[r_idx]
            row = {
                "id": f"{job_id[j_idx]}|{rid}",
                "resume_text": res_plain[r_idx],
                "resume_text_withname": res_withname[r_idx],
                "job_desc": job_text[j_idx],
                "name": res_name[r_idx],
                "group": res_group[r_idx],
                "label": 1,
            }
            if args.add_scores:
                row["sim_score"] = float(sim_row[r_idx])
                row["rank"] = rank_pos
            rows.append(row)

        # negatives
        num_negs = min(len(neg_pool), len(pos_idx) * args.neg_per_pos)
        if num_negs > 0:
            chosen = rng.choice(neg_pool, size=num_negs, replace=False)
            for r_idx in chosen:
                rid = res_id[r_idx]
                row = {
                    "id": f"{job_id[j_idx]}|{rid}",
                    "resume_text": res_plain[r_idx],
                    "resume_text_withname": res_withname[r_idx],
                    "job_desc": job_text[j_idx],
                    "name": res_name[r_idx],
                    "group": res_group[r_idx],
                    "label": 0,
                }
                if args.add_scores:
                    row["sim_score"] = float(sim_row[r_idx])
                    row["rank"] = None
                rows.append(row)

    out_cols = ["id","resume_text","resume_text_withname","job_desc","name","group","label"]
    if args.add_scores:
        out_cols += ["sim_score","rank"]

    out_df = pd.DataFrame(rows, columns=out_cols)

    if args.shuffle:
        out_df = out_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    out_df.to_csv(args.output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Done. wrote {len(out_df)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
