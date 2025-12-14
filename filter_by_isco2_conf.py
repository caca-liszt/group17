# filter_by_isco2_conf.py
# ------------------------------------------------------------
# ISCO2 + proxy confidence filtering (no name injection yet)
# Inputs:
#   - Resume.csv  (must contain: ID, Resume_str; optional: Resume_html, Category)
#   - JD_data.csv (must contain: ISCO, description; optional: JD_ID, other metadata)
# Outputs:
#   - resume_assignment_confidence_isco2.csv  (per-resume ISCO2 + proxy_conf)
#   - resumes_filtered_isco2_conf060.csv
#   - jds_filtered_isco2_conf060.csv
#   - isco2_coverage_table_conf060.csv
#   - filter_before_after_summary.csv
#   - resume_index_by_isco2.csv
#   - jd_index_by_isco2.csv
# Params (tweakable at top): K (neighbors), THR (confidence threshold), MIN_PER_GROUP (>=20/20 rule)
# ------------------------------------------------------------
# pip install scikit-learn pandas numpy

import re
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# Hyperparameters
# =========================
K = 5                 # Top-K nearest JDs
THR = 0.60            # proxy_conf threshold
MIN_PER_GROUP = 20    # keep ISCO2 groups with >=20 resumes & >=20 JDs


# =========================
# Helpers
# =========================
def isco2(x: str | int | float | None) -> str | None:
    """Extract 2-digit ISCO major group from any ISCO-like field."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = re.sub(r"[^\d]", "", str(x))
    return s[:2] if len(s) >= 2 else None


def load_inputs(resume_path: str = "Resume.csv", jd_path: str = "JD_data.csv"):
    r = pd.read_csv(resume_path)
    j = pd.read_csv(jd_path)

    # Basic schema checks
    if not {"ID", "Resume_str"}.issubset(r.columns):
        raise ValueError(f"Resume.csv must have columns: ID, Resume_str. Got: {list(r.columns)}")
    if not {"ISCO", "description"}.issubset(j.columns):
        raise ValueError(f"JD_data.csv must have columns: ISCO, description. Got: {list(j.columns)}")

    # Normalize ISCO2 on JD side
    j = j.copy()
    j["ISCO2"] = j["ISCO"].apply(isco2)
    j = j.dropna(subset=["ISCO2"]).copy()

    # Ensure JD_ID
    if "JD_ID" not in j.columns:
        j = j.reset_index(drop=False).rename(columns={"index": "JD_ID"})
    j["JD_ID"] = j["JD_ID"].astype(str)

    return r, j


def build_tfidf_embeddings(resumes: pd.DataFrame, jds: pd.DataFrame):
    """TF-IDF(1–2gram, unicode) on Resume_str + JD description."""
    resume_texts = resumes["Resume_str"].fillna("")
    jd_texts = jds["description"].fillna("")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2
    )
    X_res = vectorizer.fit_transform(resume_texts)
    X_jd = vectorizer.transform(jd_texts)
    return vectorizer, X_res, X_jd


def assign_isco2_with_proxy_conf(resumes: pd.DataFrame, jds: pd.DataFrame, X_res, X_jd,
                                 k: int = 5, batch: int = 200) -> pd.DataFrame:
    """
    For each resume:
      - take Top-K nearest JDs by cosine
      - sum similarities per JD's ISCO2
      - winner class has max summed similarity
      - proxy_conf = (sum over winner class) / (sum over Top-K)
    Returns: DataFrame[ID, ISCO2, proxy_conf]
    """
    jd_labels = jds["ISCO2"].reset_index(drop=True)

    assignments = []
    for start in range(0, X_res.shape[0], batch):
        end = min(X_res.shape[0], start + batch)
        sims = cosine_similarity(X_res[start:end], X_jd)  # (batch, n_jd)

        # Get Top-K indices per row
        k_eff = min(k, sims.shape[1]) if sims.shape[1] > 0 else 1
        topk_idx = np.argpartition(-sims, kth=k_eff - 1, axis=1)[:, :k_eff]
        for i in range(end - start):
            idxs = topk_idx[i]
            # sort by similarity descending
            order = np.argsort(-sims[i, idxs])
            idxs = idxs[order]
            svals = sims[i, idxs]
            labs = jd_labels.iloc[idxs].tolist()

            # sum sims per class
            sums = {}
            for lab, s in zip(labs, svals):
                sums[lab] = sums.get(lab, 0.0) + float(s)

            if not sums:
                assignments.append((None, np.nan))
                continue

            lab_win, sum_win = max(sums.items(), key=lambda t: t[1])
            sum_all = float(np.sum(svals)) if len(svals) else 1.0
            proxy_conf = sum_win / (sum_all + 1e-12)
            assignments.append((lab_win, proxy_conf))

    out = pd.DataFrame(assignments, columns=["ISCO2", "proxy_conf"])
    out.insert(0, "ID", resumes["ID"].values)
    return out


def apply_filters(resumes: pd.DataFrame, jds: pd.DataFrame, assign_df: pd.DataFrame,
                  thr: float = 0.60, min_per_group: int = 20):
    """
    1) Keep resumes with proxy_conf >= thr
    2) Keep only ISCO2 groups that have >= min_per_group resumes and JDs
    Returns: (resumes_final, jds_final, coverage_df, before_after_summary_df)
    """
    # Merge
    r_conf = resumes.merge(assign_df, on="ID", how="left")
    r_conf = r_conf[r_conf["proxy_conf"] >= thr].copy()

    # Counts per ISCO2
    r_counts = r_conf["ISCO2"].value_counts()
    j_counts = jds["ISCO2"].value_counts()

    kept_isco2 = sorted(
        set(r_counts.index[r_counts >= min_per_group]).intersection(
            set(j_counts.index[j_counts >= min_per_group])
        )
    )

    r_final = r_conf[r_conf["ISCO2"].isin(kept_isco2)].copy()
    j_final = jds[jds["ISCO2"].isin(kept_isco2)].copy()

    coverage = pd.DataFrame({
        "ISCO2": kept_isco2,
        "num_resumes_after_conf": [int(r_counts.get(k, 0)) for k in kept_isco2],
        "num_jds": [int(j_counts.get(k, 0)) for k in kept_isco2],
        "conf_threshold": [thr] * len(kept_isco2),
        "topk": [K] * len(kept_isco2)
    }).sort_values(["num_resumes_after_conf", "num_jds"], ascending=False)

    before_after = pd.DataFrame([{
        "resumes_before": int(len(resumes)),
        "jds_before": int(len(jds)),
        "resumes_after_conf_filter": int(len(r_final)),
        "jds_after_filter": int(len(j_final)),
        "num_isco2_kept": int(len(kept_isco2)),
        "conf_threshold": thr,
        "topk": K,
        "min_per_group": min_per_group
    }])

    return r_final, j_final, coverage, before_after


def main():
    print(">> Loading inputs …")
    resumes, jds = load_inputs()

    print(">> Building TF-IDF and embeddings …")
    _, X_res, X_jd = build_tfidf_embeddings(resumes, jds)

    print(f">> Assigning ISCO2 with Top-{K} and computing proxy confidence …")
    assign_df = assign_isco2_with_proxy_conf(resumes, jds, X_res, X_jd, k=K)
    assign_df.to_csv("resume_assignment_confidence_isco2.csv", index=False)

    print(f">> Applying filters: proxy_conf >= {THR}, and >= {MIN_PER_GROUP}/{MIN_PER_GROUP} per ISCO2 …")
    r_final, j_final, coverage, before_after = apply_filters(
        resumes, jds, assign_df, thr=THR, min_per_group=MIN_PER_GROUP
    )

    print(">> Saving outputs …")
    r_final.to_csv("resumes_filtered_isco2_conf060.csv", index=False)
    j_final.to_csv("jds_filtered_isco2_conf060.csv", index=False)
    coverage.to_csv("isco2_coverage_table_conf060.csv", index=False)
    before_after.to_csv("filter_before_after_summary.csv", index=False)

    # convenience indices for downstream “within-occupation” retrieval
    r_idx = r_final[["ID", "ISCO2"]]
    j_idx = j_final[["JD_ID", "ISCO2"]]
    r_idx.to_csv("resume_index_by_isco2.csv", index=False)
    j_idx.to_csv("jd_index_by_isco2.csv", index=False)

    print(">> Done.")
    print(before_after.to_string(index=False))


if __name__ == "__main__":
    main()
