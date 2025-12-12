# make_views_and_inject.py
# ------------------------------------------------------------
# Build Full & Title-only views, then inject names using:
#   Scheme A = "first name + per-group small surname pool"
#
# Inputs:
#   - resumes_filtered_isco2_conf060.csv  (must have: ID, ISCO2, Resume_str; optional: Category)
#   - jds_filtered_isco2_conf060.csv      (not modified, kept for pipeline completeness)
#   - names_first_only_per_group_pool.csv (group, gender, first_name, last_name)
#
# Outputs:
#   - resumes_full_no_name.csv
#   - resumes_title_only_no_name.csv
#   - resumes_full_injected.csv
#   - resumes_title_only_injected.csv
#   - variants_meta.csv
#
# Usage:
#   python make_views_and_inject.py
#
# Tweakables:
#   GROUPS = ["white","black","east_asian","south_asian_indian"]
#   EMAIL_DOMAIN = "apply.example.org"
#   GENDER_MODE = "alternate"  # or "random"
#   RNG_SEED = 2025
# ------------------------------------------------------------

import csv
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# ----------------- settings -----------------
RESUMES_PATH = "resumes_filtered_isco2_conf060.csv"
JDS_PATH      = "jds_filtered_isco2_conf060.csv"  # not used directly but kept for completeness
NAMES_PATH    = "names_first_only_per_group_pool.csv"

OUT_FULL_NO   = "resumes_full_no_name.csv"
OUT_TITL_NO   = "resumes_title_only_no_name.csv"
OUT_FULL_INJ  = "resumes_full_injected.csv"
OUT_TITL_INJ  = "resumes_title_only_injected.csv"
OUT_META      = "variants_meta.csv"

GROUPS        = ["white", "black", "east_asian", "south_asian_indian"]
EMAIL_DOMAIN  = "apply.example.org"
GENDER_MODE   = "alternate"   # "alternate" or "random"
RNG_SEED      = 2025
# --------------------------------------------

def slugify(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9]+", ".", s).strip(".").lower()
    return s

def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    r = pd.read_csv(RESUMES_PATH)
    j = pd.read_csv(JDS_PATH)
    names = pd.read_csv(NAMES_PATH)
    needed_r = {"ID","ISCO2","Resume_str"}
    if not needed_r.issubset(r.columns):
        raise ValueError(f"resumes file must contain columns {needed_r}, got {list(r.columns)}")
    needed_n = {"group","gender","first_name","last_name"}
    if not needed_n.issubset(names.columns):
        raise ValueError(f"names file must contain columns {needed_n}, got {list(names.columns)}")
    return r, j, names

def build_views(resumes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Title 取自 Category；没有则用 "Candidate"
    title_series = resumes["Category"] if "Category" in resumes.columns else pd.Series(["Candidate"]*len(resumes))
    title_series = title_series.fillna("Candidate").astype(str)

    full_no = resumes[["ID","ISCO2","Resume_str"]].copy().rename(columns={"Resume_str":"text_full"})
    full_no.insert(2, "Title", title_series.values)

    title_no = resumes[["ID","ISCO2"]].copy()
    title_no["Title"] = title_series.values
    title_no["text_title_only"] = title_no["Title"]
    return full_no, title_no

def make_pool(names: pd.DataFrame) -> Dict[str, Dict[str, pd.DataFrame]]:
    pool = {}
    for g in GROUPS:
        sub = names[names["group"]==g]
        pool[g] = {
            "male":   sub[sub["gender"]=="male"].reset_index(drop=True),
            "female": sub[sub["gender"]=="female"].reset_index(drop=True)
        }
        if pool[g]["male"].empty or pool[g]["female"].empty:
            raise ValueError(f"names pool for group '{g}' missing male or female rows")
    return pool

def next_name(pool_map, idx_map, g: str, sex: str):
    dfp = pool_map[g][sex]
    i = idx_map[(g,sex)] % len(dfp)
    idx_map[(g,sex)] += 1
    row = dfp.iloc[i]
    return row.first_name, row.last_name

def pick_gender(i: int, rng: np.random.RandomState) -> str:
    if GENDER_MODE == "alternate":
        return "male" if i % 2 == 0 else "female"
    else:
        return "male" if rng.rand() < 0.5 else "female"

def inject_full(text_full: str, first: str, last: str) -> Tuple[str,str,str]:
    full_name = f"{first} {last}"
    email = f"{slugify(first+'.'+last)}@{EMAIL_DOMAIN}"
    injected = f"{full_name}\nEmail: {email}\n\n{text_full}"
    return injected, full_name, email

def inject_title_only(title: str, first: str, last: str) -> Tuple[str,str,str]:
    full_name = f"{first} {last}"
    email = f"{slugify(first+'.'+last)}@{EMAIL_DOMAIN}"
    injected = f"{full_name}\nEmail: {email}\n\n{title}"
    return injected, full_name, email

def main():
    rng = np.random.RandomState(RNG_SEED)

    resumes, jds, names = load_inputs()
    full_no, title_no = build_views(resumes)
    full_no.to_csv(OUT_FULL_NO, index=False)
    title_no.to_csv(OUT_TITL_NO, index=False)

    pool_map = make_pool(names)
    cycle_idx = {(g,sex): 0 for g in GROUPS for sex in ["male","female"]}

    rows_full, rows_title, meta = [], [], []

    for i, r in resumes.iterrows():
        base_id = str(r["ID"])
        isco2 = str(r["ISCO2"])
        title = str(title_no.loc[title_no["ID"]==r["ID"], "Title"].values[0]) if "Title" in title_no.columns else "Candidate"
        text_full = str(r["Resume_str"])

        for g in GROUPS:
            sex = pick_gender(i, rng)
            fn, ln = next_name(pool_map, cycle_idx, g, sex)
            # Full
            inj_full, fullname, email = inject_full(text_full, fn, ln)
            id_full = f"{base_id}_{g}_{sex}_{slugify(fn+'_'+ln)}_full"
            rows_full.append({
                "ID_variant": id_full,
                "ID_base": base_id,
                "group": g,
                "gender": sex,
                "ISCO2": isco2,
                "first_name": fn,
                "last_name": ln,
                "full_name": fullname,
                "email": email,
                "text_full_injected": inj_full
            })
            meta.append({
                "ID_variant": id_full, "view":"full", "ID_base": base_id, "group": g,
                "gender": sex, "first_name": fn, "last_name": ln, "email": email, "ISCO2": isco2
            })
            # Title-only
            inj_title, fullname2, email2 = inject_title_only(title, fn, ln)
            id_title = f"{base_id}_{g}_{sex}_{slugify(fn+'_'+ln)}_title"
            rows_title.append({
                "ID_variant": id_title,
                "ID_base": base_id,
                "group": g,
                "gender": sex,
                "ISCO2": isco2,
                "first_name": fn,
                "last_name": ln,
                "full_name": fullname2,
                "email": email2,
                "text_title_only_injected": inj_title
            })
            meta.append({
                "ID_variant": id_title, "view":"title_only", "ID_base": base_id, "group": g,
                "gender": sex, "first_name": fn, "last_name": ln, "email": email2, "ISCO2": isco2
            })

    df_full_inj = pd.DataFrame(rows_full)
    df_title_inj = pd.DataFrame(rows_title)
    df_meta = pd.DataFrame(meta)

    df_full_inj.to_csv(OUT_FULL_INJ, index=False)
    df_title_inj.to_csv(OUT_TITL_INJ, index=False)
    df_meta.to_csv(OUT_META, index=False)

    print("Done.")
    print(f"Saved: {OUT_FULL_NO}, {OUT_TITL_NO}, {OUT_FULL_INJ}, {OUT_TITL_INJ}, {OUT_META}")

if __name__ == "__main__":
    main()
