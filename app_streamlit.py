import os, json
from typing import Tuple, Dict, Any, Optional, List
import pandas as pd
import streamlit as st

ART_DIR = "artifacts"
MAX_ROWS = 1000  # cap table for speed

# ---------- Data loaders ----------

@st.cache_data(show_spinner=False)
def load_mode(mode: str) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]], Optional[str]]:
    assert mode in ("pre", "post")
    csv_path = os.path.join(ART_DIR, f"{mode}pay.csv")
    json_path = os.path.join(ART_DIR, f"{mode}pay.json")
    if not (os.path.isfile(csv_path) and os.path.isfile(json_path)):
        return None, None, None
    df = pd.read_csv(csv_path)
    # Make sure some columns are present even if empty in this run
    for col in ["anomaly_type","rag_path","rag_score","rag_snippet","fee_allowed","contract_tier",
                "eligibility_start","eligibility_end","pa_required","has_auth"]:
        if col not in df.columns:
            df[col] = pd.NA
    # Friendly column order
    prefer = [
        "anomaly_type","claim_id","member_id","provider_id","service_date","lob",
        "cpt_code","units","place_of_service",
        "paid_amount","allowed_amount","billed_amount",
        "eligibility_start","eligibility_end","contract_tier","fee_schedule_id","fee_allowed",
        "pa_required","has_auth","in_network_flag",
        "rag_path","rag_score","rag_snippet"
    ]
    cols = [c for c in prefer if c in df.columns] + [c for c in df.columns if c not in prefer]
    df = df[cols]

    with open(json_path, "r") as f:
        summary = json.load(f)

    # light typing
    for c in ("service_date","eligibility_start","eligibility_end"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.date

    return df, summary, csv_path

def anomaly_rollup_from_json(summary: Dict[str, Any]) -> pd.DataFrame:
    rows = summary.get("anomalies", []) or []
    if not rows:
        return pd.DataFrame(columns=["type","count","paid","allowed"])
    return pd.DataFrame(rows)

# ---------- UI helpers ----------

def header_metrics(summary: Dict[str, Any], mode: str):
    st.subheader(f"{'Pre-pay' if mode=='pre' else 'Post-pay'} worklist")
    if not summary:
        st.info("No artifacts found yet. Run `reconcile.py` to generate outputs.")
        return
    period = summary["period"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Claims", f"{summary['totals']['claims']:,}")
    c2.metric("Members", f"{summary['totals']['members']:,}")
    c3.metric("Paid $", f"{summary['totals']['paid']:,.2f}")
    c4.metric("Allowed $", f"{summary['totals']['allowed']:,.2f}")
    st.caption(f"Period: {period['start_date']} → {period['end_date']}  |  LOB: {summary['lob']}")

def anomaly_bar(roll: pd.DataFrame):
    if roll is None or roll.empty:
        st.write("No anomalies.")
        return
    roll2 = roll.sort_values("count", ascending=False).head(12)
    roll2 = roll2.rename(columns={"type":"anomaly"})
    st.bar_chart(roll2.set_index("anomaly")["count"])

def filter_block(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.header("Filters")
        mode = st.session_state.get("mode", "pre")
        st.caption("Switch mode at the top of the page.")

        # anomaly filter
        anomalies = sorted(df["anomaly_type"].dropna().unique().tolist()) if "anomaly_type" in df.columns else []
        sel_anoms = st.multiselect("Anomaly type", anomalies, default=[])
        # text search
        q = st.text_input("Search (member/provider/CPT/claim)")
        # numeric filter (paid)
        min_paid, max_paid = float(df["paid_amount"].min() or 0), float(df["paid_amount"].max() or 0)
        paid_range = st.slider("Paid amount range", min_value=float(round(min_paid,2)),
                               max_value=float(round(max_paid,2)) if max_paid>0 else 0.0,
                               value=(float(round(min_paid,2)), float(round(max_paid,2))) if max_paid>0 else (0.0,0.0))
        # sort
        options = [c for c in ["paid_amount","allowed_amount","service_date","member_id","provider_id","cpt_code","anomaly_type"] if c in df.columns]
        sort_by = st.selectbox("Sort by", options, index=0 if options else 0)
        sort_dir = st.radio("Direction", ["desc","asc"], horizontal=True, index=0)
    view = df.copy()

    if sel_anoms:
        view = view[view["anomaly_type"].isin(sel_anoms)]

    if q:
        ql = q.lower()
        keys = [k for k in ["member_id","provider_id","cpt_code","claim_id"] if k in view.columns]
        mask = False
        for k in keys:
            mask = mask | view[k].astype(str).str.lower().str.contains(ql, na=False)
        view = view[mask]

    if "paid_amount" in view.columns and paid_range != (0.0, 0.0):
        lo, hi = paid_range
        view = view[(view["paid_amount"].fillna(0.0) >= lo) & (view["paid_amount"].fillna(0.0) <= hi)]

    if sort_by in view.columns:
        view = view.sort_values(sort_by, ascending=(sort_dir=="asc"))

    return view

def detail_panel(view: pd.DataFrame):
    if view.empty: 
        st.info("No rows after filters.")
        return
    st.markdown("### Table")
    st.caption(f"Showing up to {MAX_ROWS:,} rows (of {len(view):,} after filters).")
    st.dataframe(view.head(MAX_ROWS), use_container_width=True, hide_index=True)

    # Simple detail selector
    st.markdown("### Claim detail")
    claim_ids: List[str] = view["claim_id"].astype(str).tolist()
    chosen = st.selectbox("Pick a claim_id", claim_ids[:200])  # limit options for usability
    rec = view[view["claim_id"].astype(str) == chosen].iloc[0].to_dict()

    cols_a = ["anomaly_type","claim_id","member_id","provider_id","service_date","lob","place_of_service"]
    cols_b = ["cpt_code","units","paid_amount","allowed_amount","billed_amount","fee_allowed","contract_tier"]
    cols_c = ["eligibility_start","eligibility_end","pa_required","has_auth","in_network_flag"]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("**Core**")
        for k in cols_a:
            if k in rec: st.write(f"- {k}: `{rec.get(k)}`")
    with c2:
        st.write("**Financial/Codes**")
        for k in cols_b:
            if k in rec: st.write(f"- {k}: `{rec.get(k)}`")
    with c3:
        st.write("**Context**")
        for k in cols_c:
            if k in rec: st.write(f"- {k}: `{rec.get(k)}`")

    if rec.get("rag_snippet"):
        with st.expander("RAG snippet (investigative context)"):
            st.code(str(rec["rag_snippet"]))
            st.caption(f"Source: {rec.get('rag_path','(unknown)')}  |  score: {rec.get('rag_score')}")

# ---------- App ----------

st.set_page_config(page_title="Back Office Worklists", layout="wide")

st.title("Back Office Worklists (Local)")
st.caption("Browse deterministic pre-pay and investigative post-pay flags generated by `reconcile.py`.")

# top-level mode switch
mode = st.segmented_control("Mode", options=["pre","post"], default="pre", key="mode")

df, summary, csv_path = load_mode(mode)
if df is None or summary is None:
    st.warning(f"No `{mode}pay.csv/json` found in `{ART_DIR}`.\n\nRun:\n\n```bash\npython reconcile.py 2025-07-01 2025-07-31 COMM --mode {mode} [--rag-dir rag_policies]\n```")
    st.stop()

header_metrics(summary, mode)

col_a, col_b = st.columns([2,3], gap="large")

with col_a:
    st.markdown("### Anomaly breakdown")
    roll = anomaly_rollup_from_json(summary)
    anomaly_bar(roll)
    st.markdown("### Downloads")
    json_name = f"{mode}pay.json"
    with open(os.path.join(ART_DIR, json_name), "rb") as fh:
        st.download_button("Download summary JSON", data=fh, file_name=json_name, mime="application/json")
    with open(os.path.join(ART_DIR, f"{mode}pay.csv"), "rb") as fh:
        st.download_button("Download worklist CSV", data=fh, file_name=f"{mode}pay.csv", mime="text/csv")

with col_b:
    view = filter_block(df)
    detail_panel(view)
