"""Merge the two blind reviews (A, B) of testset_v1 and list the rows that need a third reviewer.

Usage: python scripts/testset_merge.py
Reads  sites/testset_v1/labels/{A,B}_batch_NN.csv
Writes sites/testset_v1/merge/ab_merged.csv (one row per sample: answer_A, answer_B, agree, third_needed, reason)
       sites/testset_v1/merge/third_list.csv (sample ids for the third reviewer)
Never reads key/.
"""
import glob, os, sys
import pandas as pd

ROOT = "sites/testset_v1"
# rows where a reviewer hit the image limit and relied on earlier notes (from the agents' hand-backs)
LOWCONF = {
    "B": ["T041", "T042"] + [f"T{i:03d}" for i in range(53, 61)] + [f"T{i:03d}" for i in range(163, 180)] + ["T244", "T245", "T247", "T248", "T250"]
         + [f"T{i:03d}" for i in list(range(306, 312)) + [313, 314] + list(range(316, 321))],
    "A": [f"T{i:03d}" for i in range(45, 63)],
}

def load(rev):
    fs = sorted(glob.glob(f"{ROOT}/labels/{rev}_batch_*.csv"))
    df = pd.concat([pd.read_csv(f, dtype=str).assign(batch=os.path.basename(f)) for f in fs])
    df["sample_id"] = df["sample_id"].str.strip()
    df["answer"] = df["answer"].str.strip().str.lower()
    bad = ~df["answer"].isin(["cane", "not_cane", "unsure"])
    if bad.any():
        print("BAD answers:", df[bad][["batch", "sample_id", "answer"]].to_string()); sys.exit(1)
    dup = df["sample_id"].duplicated()
    if dup.any():
        print("DUPLICATES:", df[dup]["sample_id"].tolist()); sys.exit(1)
    return df.set_index("sample_id")

A, B = load("A"), load("B")
ids = [f"T{i:03d}" for i in range(1, 481)]
m = pd.DataFrame(index=ids)
for rev, df in (("A", A), ("B", B)):
    for c in ("answer", "what", "confidence", "note"):
        m[f"{c}_{rev}"] = df[c].reindex(ids)
    m[f"lowconf_{rev}"] = m.index.isin(LOWCONF[rev])
missing = m[m["answer_A"].isna() | m["answer_B"].isna()].index.tolist()
print(f"missing reviews: {len(missing)}", missing[:10])
m["agree"] = m["answer_A"] == m["answer_B"]
def reason(r):
    if pd.isna(r.answer_A) or pd.isna(r.answer_B): return "missing"
    if r.answer_A == "unsure" or r.answer_B == "unsure": return "unsure"
    if r.answer_A != r.answer_B: return "disagree"
    if r.lowconf_A or r.lowconf_B: return "lowconf"
    if "low" in str(r.confidence_A) and "low" in str(r.confidence_B): return "both_low"
    return ""
m["reason"] = m.apply(reason, axis=1)
m["third_needed"] = m["reason"].isin(["unsure", "disagree", "lowconf", "both_low"])
m.index.name = "sample_id"
m.to_csv(f"{ROOT}/merge/ab_merged.csv")
third = m[m["third_needed"]]
third[["reason"]].to_csv(f"{ROOT}/merge/third_list.csv")
print("answers A:", m.answer_A.value_counts().to_dict())
print("answers B:", m.answer_B.value_counts().to_dict())
both = m.dropna(subset=["answer_A", "answer_B"])
print(f"agreement (both reviewed, n={len(both)}): {both.agree.mean():.3f}")
print("third reviewer reasons:", third.reason.value_counts().to_dict(), "total", len(third))
