import os, sys, json, hashlib, sqlite3, uuid, argparse
from datetime import datetime
import pandas as pd
from jsonschema import validate

DB_PATH = "backoffice.db"
OUTPUT_DIR = "artifacts"
AMOUNT_RATIO = 1.25

RECON_SCHEMA = {
  "type": "object",
  "required": ["run_id","mode","period","lob","totals","anomalies","artifacts","generated_at"],
  "properties": {
    "run_id": {"type":"string"},
    "mode": {"type":"string"},
    "period": {"type":"object","required":["start_date","end_date"],
               "properties":{"start_date":{"type":"string"}, "end_date":{"type":"string"}}},
    "lob": {"type":"string"},
    "totals": {"type":"object","required":["claims","members","paid","allowed"],
               "properties":{"claims":{"type":"integer"},"members":{"type":"integer"},
                             "paid":{"type":"number"},"allowed":{"type":"number"}}},
    "anomalies": {"type":"array","items":{
        "type":"object",
        "required":["type","count","paid","allowed"],
        "properties":{"type":{"type":"string"},
                      "count":{"type":"integer"},
                      "paid":{"type":"number"},
                      "allowed":{"type":"number"}}
    }},
    "artifacts": {"type":"object","required":["csv_path","sql_fingerprints"],
                  "properties":{"csv_path":{"type":"string"},
                                "sql_fingerprints":{"type":"array","items":{"type":"string"}}}},
    "generated_at": {"type":"string"}
  }
}

def sql_hash(q: str) -> str: return hashlib.sha256(q.encode()).hexdigest()[:16]
def fetch_df(conn, q, params=()): return pd.read_sql_query(q, conn, params=params)

# ---------------- Tiny local RAG ----------------
class TinyRAG:
    def __init__(self, rag_dir: str|None):
        self.docs, self.paths, self.vectorizer, self.matrix, self.sim = [], [], None, None, None
        if not rag_dir or not os.path.isdir(rag_dir): return
        for root,_,files in os.walk(rag_dir):
            for f in files:
                if f.lower().endswith((".txt",".md")):
                    p = os.path.join(root,f)
                    with open(p,"r",encoding="utf-8",errors="ignore") as fh:
                        text = fh.read()
                    for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
                        self.docs.append(para); self.paths.append(p)
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import linear_kernel
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.matrix = self.vectorizer.fit_transform(self.docs) if self.docs else None
            self.sim = lambda q: linear_kernel(self.vectorizer.transform([q]), self.matrix)[0]
        except Exception:
            self.vectorizer = None; self.matrix = None; self.sim = None

    def search(self, query: str, topk: int = 3):
        if not self.docs: return []
        if self.sim:
            scores = self.sim(query)
            idxs = scores.argsort()[::-1][:topk]
            return [(self.docs[i], self.paths[i], float(scores[i])) for i in idxs if scores[i] > 0]
        # fallback: naive keywords
        q = query.lower(); scored = []
        for i,txt in enumerate(self.docs):
            score = sum(tok in txt.lower() for tok in set(q.split()))
            if score: scored.append((txt, self.paths[i], float(score)))
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:topk]

# ---------------- Main ----------------
def main(start_date: str, end_date: str, lob: str, mode: str = "both", rag_dir: str|None = None):
    conn = sqlite3.connect(DB_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rag = TinyRAG(rag_dir)

    # Pull facts
    Q_CLAIMS = """
      SELECT claim_id, member_id, provider_id, service_date, place_of_service,
             billed_amount, allowed_amount, paid_amount, in_network_flag,
             cpt_code, diagnosis_code, units, lob
      FROM claims
      WHERE service_date BETWEEN ? AND ? AND lob = ?
    """
    claims = fetch_df(conn, Q_CLAIMS, (start_date, end_date, lob))
    claims["service_date"] = pd.to_datetime(claims["service_date"])

    member_ids = tuple(claims["member_id"].unique().tolist() or ["__none__"])
    Q_ELIG = f"""
      SELECT member_id, eligibility_start, eligibility_end, product, network_tier
      FROM eligibility
      WHERE member_id IN ({",".join(["?"]*len(member_ids))})
    """
    elig = fetch_df(conn, Q_ELIG, member_ids)
    elig["eligibility_start"] = pd.to_datetime(elig["eligibility_start"])
    elig["eligibility_end"] = pd.to_datetime(elig["eligibility_end"])

    Q_CONTRACTS = "SELECT * FROM provider_contracts"
    contracts = fetch_df(conn, Q_CONTRACTS)
    contracts["start_date"] = pd.to_datetime(contracts["start_date"])
    contracts["end_date"] = pd.to_datetime(contracts["end_date"])

    Q_FS = "SELECT * FROM fee_schedules"
    fee = fetch_df(conn, Q_FS)
    fee["start_date"] = pd.to_datetime(fee["start_date"])
    fee["end_date"] = pd.to_datetime(fee["end_date"])

    Q_POLICY = "SELECT * FROM policy_rules"
    policy = fetch_df(conn, Q_POLICY)
    policy["start_date"] = pd.to_datetime(policy["start_date"])
    policy["end_date"] = pd.to_datetime(policy["end_date"])

    Q_AUTH = "SELECT * FROM authorisations"
    auth = fetch_df(conn, Q_AUTH)
    auth["start_date"] = pd.to_datetime(auth["start_date"])
    auth["end_date"] = pd.to_datetime(auth["end_date"])

    # Eligibility containment (effective-dated)
    cl = claims.merge(elig, on="member_id", how="left", suffixes=("","_elig"))
    mask = (cl["service_date"]>=cl["eligibility_start"]) & (cl["service_date"]<=cl["eligibility_end"])
    good = cl[mask]
    best = good.sort_values(["claim_id","eligibility_start"]).drop_duplicates("claim_id", keep="last")
    missing = claims[~claims["claim_id"].isin(best["claim_id"])]
    cl_elig = pd.concat([
        best[["claim_id","member_id","provider_id","service_date","place_of_service","billed_amount","allowed_amount","paid_amount","in_network_flag","cpt_code","diagnosis_code","units","lob","eligibility_start","eligibility_end","product"]],
        missing.assign(eligibility_start=pd.NaT, eligibility_end=pd.NaT, product=None)
    ], ignore_index=True)

    # Provider contracts on DOS (scalar comparisons; safe)
    def contract_on_dos(df_row):
        subset = contracts[
            (contracts["provider_id"] == df_row["provider_id"]) &
            (contracts["start_date"] <= df_row["service_date"]) &
            (df_row["service_date"] <= contracts["end_date"])
        ]
        if subset.empty:
            return pd.Series({"contract_tier": None, "fee_schedule_id": None})
        row = subset.sort_values("start_date").iloc[-1]
        return pd.Series({"contract_tier": row["tier"], "fee_schedule_id": row["fee_schedule_id"]})
    cont = cl_elig.apply(contract_on_dos, axis=1)
    clx = pd.concat([cl_elig, cont], axis=1)

    # Vectorized fee schedule match on DOS → clx["fee_allowed"]
    fee_m = clx.merge(fee, on=["fee_schedule_id", "cpt_code"], how="left", suffixes=("", "_fee"))
    fee_m = fee_m[
        (fee_m["start_date"] <= fee_m["service_date"]) &
        (fee_m["service_date"] <= fee_m["end_date"])
    ]
    fee_pick = (
        fee_m.sort_values(["claim_id", "start_date"])
             .drop_duplicates("claim_id", keep="last")[["claim_id", "allowed_amount"]]
             .rename(columns={"allowed_amount": "fee_allowed"})
    )
    clx = clx.merge(fee_pick, on="claim_id", how="left")

    # Policy requirement on DOS
    def pa_required(row):
        subset = policy[
            (policy["lob"] == row["lob"]) &
            (policy["cpt_code"] == row["cpt_code"]) &
            (policy["start_date"] <= row["service_date"]) &
            (row["service_date"] <= policy["end_date"]) &
            ((policy["pos"] == "*") | (policy["pos"] == row["place_of_service"]))
        ]
        if subset.empty:
            return None
        return int(subset.sort_values("start_date").iloc[-1]["pa_required"])
    clx["pa_required"] = clx.apply(pa_required, axis=1)

    # Has matching authorisation
    def has_auth(row):
        subset = auth[
            (auth["member_id"] == row["member_id"]) &
            (auth["cpt_code"] == row["cpt_code"]) &
            (auth["start_date"] <= row["service_date"]) &
            (row["service_date"] <= auth["end_date"]) &
            (auth["lob"] == row["lob"]) &
            (auth["pos"] == row["place_of_service"])
        ]
        if subset.empty: return 0
        return 1 if subset["units"].max() >= int(row["units"]) else 0
    clx["has_auth"] = clx.apply(has_auth, axis=1)

    rows = []
    def emit(df, atype):
        if df.empty: return
        r = df.copy(); r["anomaly_type"] = atype; rows.append(r)

    # ---------------- PRE-PAY (deterministic) ----------------
    if mode in ("pre","both"):
        emit(clx[clx["eligibility_start"].isna()], "PRE_NO_ELIGIBILITY")
        emit(clx[(~clx["eligibility_end"].isna()) & (clx["service_date"] > clx["eligibility_end"])], "PRE_ELIGIBILITY_EXPIRED")
        emit(clx[(clx["in_network_flag"]==1) & (clx["contract_tier"]=="OON")], "PRE_OON_BILLED_INN")
        if "fee_allowed" not in clx.columns: clx["fee_allowed"] = pd.NA
        emit(clx[(~clx["fee_allowed"].isna()) & (abs(clx["allowed_amount"] - clx["fee_allowed"]) > 0.01)], "PRE_PRICING_MISMATCH")
        emit(clx[(clx["pa_required"]==1) & (clx["has_auth"]==0)], "PRE_MISSING_PA")
        dup_keys = ["member_id","service_date","cpt_code","units","provider_id"]
        dup = clx.duplicated(subset=dup_keys, keep=False)
        emit(clx[dup], "PRE_EXACT_DUPLICATE")

        pre_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=list(clx.columns)+["anomaly_type"])
        csv_path = os.path.join(OUTPUT_DIR, "prepay.csv")
        pre_df.to_csv(csv_path, index=False)
        pre_summary = summarise("pre", clx, pre_df, [sql_hash(Q_CLAIMS), sql_hash(Q_ELIG), sql_hash(Q_CONTRACTS), sql_hash(Q_FS), sql_hash(Q_POLICY), sql_hash(Q_AUTH)], start_date, end_date, lob)
        write_summary(pre_summary, "prepay", csv_path)

    # ---------------- POST-PAY (heuristic / investigative) ----------------
    rows = []
    if mode in ("post","both"):
        emit(clx[clx["paid_amount"] > clx["allowed_amount"] * AMOUNT_RATIO], "POST_RATIO_OUTLIER")

        gcols = ["member_id","service_date","cpt_code"]
        nd = []
        for _,grp in clx.groupby(gcols):
            if grp.shape[0] < 2: continue
            grp2 = grp.sort_values("paid_amount")
            a = grp2.iloc[0]; b = grp2.iloc[-1]
            if abs(int(a["units"])-int(b["units"]))<=1 or abs(float(a["paid_amount"])-float(b["paid_amount"]))<=5.0:
                nd.append(grp2)
        if nd: emit(pd.concat(nd), "POST_NEAR_DUPLICATE")

        # RAG-assisted suspected missing PA
        rag_hits = []
        if isinstance(rag, TinyRAG) and rag.docs:
            suspects = clx[(clx["has_auth"]==0)]
            for _,row in suspects.head(500).iterrows():
                q = f"LOB {row['lob']} CPT {row['cpt_code']} place {row['place_of_service']} prior authorization required?"
                hits = rag.search(q, topk=1)
                if not hits: continue
                para, path, score = hits[0]
                low = para.lower()
                if (row['cpt_code'] in para) and ("prior authorization" in low or "pa required" in low):
                    r = row.copy()
                    r["rag_path"] = path
                    r["rag_snippet"] = para[:400]
                    r["rag_score"] = score
                    rag_hits.append(r)
        if rag_hits:
            emit(pd.DataFrame(rag_hits), "POST_SUSPECT_MISSING_PA_RAG")

        post_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=list(clx.columns)+["anomaly_type"])
        csv_path = os.path.join(OUTPUT_DIR, "postpay.csv")
        post_df.to_csv(csv_path, index=False)
        post_summary = summarise("post", clx, post_df, [sql_hash(Q_CLAIMS), sql_hash(Q_ELIG), sql_hash(Q_CONTRACTS), sql_hash(Q_FS), sql_hash(Q_POLICY), sql_hash(Q_AUTH)], start_date, end_date, lob)
        write_summary(post_summary, "postpay", csv_path)

def summarise(mode, cl_full, flagged, sql_hashes, start_date, end_date, lob):
    totals = {
        "claims": int(cl_full.shape[0]),
        "members": int(cl_full["member_id"].nunique()),
        "paid": float(cl_full["paid_amount"].sum()),
        "allowed": float(cl_full["allowed_amount"].sum()),
    }
    agg = (flagged.groupby("anomaly_type")
           .agg(count=("claim_id","nunique"),
                paid=("paid_amount","sum"),
                allowed=("allowed_amount","sum"))
           .reset_index()) if not flagged.empty else pd.DataFrame(columns=["anomaly_type","count","paid","allowed"])
    anomalies = (agg.rename(columns={"anomaly_type":"type"})).to_dict(orient="records")

    summary = {
        "run_id": str(uuid.uuid4()),
        "mode": mode,
        "period": {"start_date": start_date, "end_date": end_date},
        "lob": lob,
        "totals": totals,
        "anomalies": anomalies,
        "artifacts": {"csv_path":"", "sql_fingerprints": sql_hashes},
        "generated_at": datetime.utcnow().isoformat() + "Z"
    }
    validate(instance=summary, schema=RECON_SCHEMA)
    return summary

def write_summary(summary, prefix, csv_path):
    summary["artifacts"]["csv_path"] = csv_path
    with open(os.path.join(OUTPUT_DIR, f"{prefix}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    lines = [
        f"{prefix.upper()} reconciliation {summary['period']['start_date']}→{summary['period']['end_date']} LOB={summary['lob']}",
        f"Totals: {summary['totals']['claims']} claims | members {summary['totals']['members']} | paid ${summary['totals']['paid']:.2f} | allowed ${summary['totals']['allowed']:.2f}",
        "Top anomalies:"
    ]
    if summary["anomalies"]:
        top = sorted(summary["anomalies"], key=lambda x: x["paid"], reverse=True)[:5]
        for a in top:
            lines.append(f"- {a['type']}: {a['count']} | paid ${a['paid']:.2f}")
    else:
        lines.append("- none")
    with open(os.path.join(OUTPUT_DIR, f"summary_{prefix}.txt"), "w") as f:
        f.write("\n".join(lines))
    print("Wrote:", csv_path, f"artifacts/{prefix}.json", f"artifacts/summary_{prefix}.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("start_date", nargs='?', default="2025-07-01")
    parser.add_argument("end_date", nargs='?', default="2025-07-31")
    parser.add_argument("lob", nargs='?', default="COMM")
    parser.add_argument("--mode", choices=["pre","post","both"], default="both")
    parser.add_argument("--rag-dir", default=None)
    args = parser.parse_args()
    main(args.start_date, args.end_date, args.lob, args.mode, args.rag_dir)
