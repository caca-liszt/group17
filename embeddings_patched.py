
import argparse
import pickle
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel

def load_lines(path):
    if path is None or path == "" or not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]

def apply_prefix(texts, prefix_path, for_query=False):
    if prefix_path is None or prefix_path == "" or not Path(prefix_path).exists():
        return texts
    prefix = Path(prefix_path).read_text(encoding="utf-8").strip()
    if not prefix:
        return texts
    # match original repo idea: append instruction to queries; prepend to docs if needed
    if for_query:
        return [f"{prefix}\n{t}" if t else prefix for t in texts]
    else:
        return [f"{prefix}\n{t}" if t else prefix for t in texts]

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

def encode_texts(texts, model, tokenizer, max_length=512, batch_size=32, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    all_embeds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            if i % (batch_size * 20) == 0:
                print(f"[progress] encoded {i}/{len(texts)}")  # <-- 每20个batch报一次
            enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            enc = {k:v.to(device) for k,v in enc.items()}
            out = model(**enc)
            if hasattr(out, "last_hidden_state"):
                emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
            elif hasattr(out, "pooler_output"):
                emb = out.pooler_output
            else:
                raise RuntimeError("Model output missing last_hidden_state/pooler_output")
            all_embeds.append(emb.cpu())
    return torch.cat(all_embeds, dim=0).numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m","--model", type=str, default="intfloat/e5-base-v2")
    ap.add_argument("-q","--queries", type=str, default=None)     # path to descriptions.txt
    ap.add_argument("-d","--documents", type=str, default=None)   # path to resumes*.txt
    ap.add_argument("-t","--task", type=str, default=None)        # task_instruction.txt
    ap.add_argument("-p","--prefixes", type=str, default=None)    # name prefixes file (we often pass "")
    ap.add_argument("-o","--out", type=str, required=True)        # output pickle path
    ap.add_argument("-l","--max_length", type=int, default=512)
    ap.add_argument("-b","--batch_size", type=int, default=32)
    args = ap.parse_args()

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModel.from_pretrained(args.model, torch_dtype=dtype, trust_remote_code=True)

    texts = []
    if args.queries:
        q = load_lines(args.queries)
        q = apply_prefix(q, args.task, for_query=True)
        texts = q
    if args.documents:
        d = load_lines(args.documents)
        d = apply_prefix(d, args.prefixes, for_query=False)
        texts = d
    if not texts:
        raise ValueError("No input texts. Provide -q or -d")

    embs = encode_texts(texts, model, tokenizer, max_length=args.max_length, batch_size=args.batch_size)
    with open(args.out, "wb") as f:
        pickle.dump(embs, f)
    print(f"Wrote embeddings: {args.out} shape={embs.shape}")

if __name__ == "__main__":
    main()
