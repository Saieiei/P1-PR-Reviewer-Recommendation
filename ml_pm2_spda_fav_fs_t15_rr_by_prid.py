#!/usr/bin/env python3
"""
ml_pm2_spda_fav_fs_t15_rr_by_prid.py
------------------------------------
• Ask for one PR-ID.
• For **each file** in that PR:
    – derive tags (via new-prs-labeler.yml, if any; else fall back to PR tags)
    – compute & print top-15 reviewers
• Export every file’s table to reviewer_scores_pr<PRID>.xlsx (one sheet / file).
• Feedback prompt at the very end.

Requirements: pandas, PyYAML, openpyxl, and ONE of
  ▸ networkx  ▸ torch-gpu  ▸ cudf + cugraph   (for PageRank)
"""

# ---------------------------------------------------------------------------#
# Imports & setup                                                            #
# ---------------------------------------------------------------------------#
import sqlite3, fnmatch, math, re, sys
from datetime import datetime

import pandas as pd
import yaml                              # PyYAML

# Optional GPU libs for fast PageRank
try:
    import cugraph, cudf                # noqa: F401
    _HAS_CUGRAPH = True
except ImportError:
    _HAS_CUGRAPH = False

try:
    import torch                        # noqa: F401
    _HAS_TORCH = torch.cuda.is_available() or getattr(torch.version, "hip", None)
except Exception:                       # torch not installed
    _HAS_TORCH = False

DB_PATH   = "pr_data.db"
YAML_PATH = "new-prs-labeler.yml"

# ---------------------------------------------------------------------------#
# Feedback helpers                                                           #
# ---------------------------------------------------------------------------#
def initialize_feedback_table(db_path=DB_PATH):
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                reviewer TEXT PRIMARY KEY,
                fav_rev_points INTEGER DEFAULT 0
            );
        """)
        conn.commit()

def get_feedback(db_path=DB_PATH):
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT reviewer, fav_rev_points FROM feedback").fetchall()
    return {r: pts for r, pts in rows}

def update_feedback_for_reviewer(reviewer, inc=1, db_path=DB_PATH):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT fav_rev_points FROM feedback WHERE reviewer=?", (reviewer,))
        row = cur.fetchone()
        if row:
            cur.execute("UPDATE feedback SET fav_rev_points=? WHERE reviewer=?",
                        (row[0] + inc, reviewer))
        else:
            cur.execute("INSERT INTO feedback VALUES (?,?)", (reviewer, inc))
        conn.commit()

# ---------------------------------------------------------------------------#
# YAML helpers                                                               #
# ---------------------------------------------------------------------------#
def _flatten(node):
    if isinstance(node, str):
        return [node]
    if isinstance(node, list):
        return sum((_flatten(n) for n in node), [])
    if isinstance(node, dict):
        return sum((_flatten(v) for v in node.values()), [])
    return []

def load_label_map(path=YAML_PATH):
    try:
        return yaml.safe_load(open(path, "r"))
    except FileNotFoundError:
        return {}

def tag_set_for_file(fp: str, label_map: dict):
    tags = set()
    for lbl, patterns in label_map.items():
        for patt in _flatten(patterns):
            if fnmatch.fnmatch(fp, patt):
                tags.add(lbl.lower())
                break
    return tags

# ---------------------------------------------------------------------------#
# PR-level helpers                                                           #
# ---------------------------------------------------------------------------#
def update_missing_pr_labels(db_path=DB_PATH, label_map=None):
    if not label_map:
        return
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT pr_id FROM pull_requests WHERE labels IS NULL OR labels=''")
        for pid, in cur.fetchall():
            cur.execute("SELECT file_path FROM pr_files WHERE pr_id=?", (pid,))
            files = [f[0] for f in cur.fetchall()]
            matched = set()
            for fp in files:
                matched.update(tag_set_for_file(fp, label_map))
            if matched:
                cur.execute("UPDATE pull_requests SET labels=? WHERE pr_id=?",
                            (", ".join(sorted(matched)), pid))
        conn.commit()

def fetch_pr_tags_files(pr_id, db_path=DB_PATH):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT labels FROM pull_requests WHERE pr_id=?", (pr_id,))
        row = cur.fetchone()
        pr_tags = {t.strip().lower() for t in row[0].split(",")} if row and row[0] else set()
        cur.execute("SELECT file_path FROM pr_files WHERE pr_id=?", (pr_id,))
        files = [f[0] for f in cur.fetchall()]
    return pr_tags, files

# ---------------------------------------------------------------------------#
# Load historical corpus                                                     #
# ---------------------------------------------------------------------------#
def load_history(db_path=DB_PATH):
    with sqlite3.connect(db_path) as conn:
        prs   = pd.read_sql_query("SELECT pr_id, labels FROM pull_requests", conn)
        files = pd.read_sql_query("SELECT pr_id, file_path FROM pr_files", conn)
        revs  = pd.read_sql_query(
            "SELECT pr_id, reviewer, review_date, state FROM reviews", conn)
    return prs, files, revs

def build_reviewer_history(prs_df, files_df, revs_df):
    file_grp = files_df.groupby("pr_id")["file_path"].apply(list).reset_index()
    prs = prs_df.merge(file_grp, on="pr_id", how="left")
    prs["labels"]    = prs["labels"].fillna("")
    prs["file_path"] = prs["file_path"].apply(lambda x: x if isinstance(x, list) else [])

    merged = revs_df.merge(prs, on="pr_id", how="left")
    hist   = {}
    for (rev, pid), g in merged.groupby(["reviewer", "pr_id"]):
        tset = {l.strip().lower() for lbls in g["labels"] for l in lbls.split(",") if l.strip()}
        fset = {fp.lower() for fps in g["file_path"] for fp in (fps if isinstance(fps, list) else [])}
        hist.setdefault(rev, {})[pid] = {"tags": tset, "files": fset}
    return hist

# ---------------------------------------------------------------------------#
# Scoring                                                                    #
# ---------------------------------------------------------------------------#
def compute_activity(revs_df, tau=30):
    revs_df["review_date"] = pd.to_datetime(revs_df["review_date"], utc=True, errors="coerce")
    now = pd.Timestamp.utcnow()
    def pts(state): return 1 if str(state).upper() in {"APPROVED","CHANGES_REQUESTED","COMMENTED","COMMIT"} else 0
    scores={}
    for _,r in revs_df.iterrows():
        if pd.isnull(r["reviewer"]) or pd.isnull(r["review_date"]): continue
        decay=math.exp(-((now-r["review_date"]).total_seconds()/86400)/tau)
        scores[r["reviewer"]] = scores.get(r["reviewer"],0)+pts(r["state"])*decay
    if scores:
        m=max(scores.values())
        for k in scores: scores[k]/=m
    return scores

def _abs_score(tags, fp, hist, wt=1.0, wf=2.0):
    s=0.0
    for d in hist.values():
        s += wt*sum(1 for t in tags if any(t in ot for ot in d["tags"])) \
           + wf*(1 if fp in d["files"] else 0)
    return s

def rank_reviewers(tags, fp, hist, pr, act, fb_w=1e-3, wt=1, wf=2, main=1, pr_w=0.2):
    raw = {r:_abs_score(tags,fp,d,wt,wf) for r,d in hist.items()}
    max_raw=max(raw.values()) if raw else 0
    fb = get_feedback(); max_fb=max(fb.values()) if fb else 0
    rows=[]
    for r,ra in raw.items():
        rows.append((r,ra,
                     ra/max_raw if max_raw else 0,
                     act.get(r,0), pr.get(r,0),
                     fb.get(r,0), fb.get(r,0)/max_fb if max_fb else 0,
                     main*((ra/max_raw if max_raw else 0)*act.get(r,0))+pr_w*pr.get(r,0)+fb_w*(fb.get(r,0)/max_fb if max_fb else 0)))
    rows.sort(key=lambda x:x[7],reverse=True)
    return rows

# ---------------------------------------------------------------------------#
# PageRank engines                                                            #
# ---------------------------------------------------------------------------#
def pagerank_cugraph(revs_df):
    rv=revs_df["reviewer"].dropna().unique().tolist()
    if not rv: return {}
    rid={r:i for i,r in enumerate(rv)}; src,dst=[],[]
    for reviewers in revs_df.groupby("pr_id")["reviewer"].apply(set).values:
        lst=list(reviewers)
        for i in range(len(lst)):
            for j in range(i+1,len(lst)):
                a,b=rid[lst[i]],rid[lst[j]]; src+=[a,b]; dst+=[b,a]
    g=cugraph.Graph(directed=False)
    g.from_cudf_edgelist(cudf.DataFrame({"src":src,"dst":dst}),
                         source="src", destination="dst", renumber=False)
    pr=g.pagerank()
    return {rv[int(row.vertex)]:float(row.pagerank) for row in pr.itertuples()}

def pagerank_torch(revs_df):
    rv=revs_df["reviewer"].dropna().unique().tolist()
    if not rv: return {}
    rid={r:i for i,r in enumerate(rv)}; edges=[]
    for reviewers in revs_df.groupby("pr_id")["reviewer"].apply(set).values:
        lst=list(reviewers)
        for i in range(len(lst)):
            for j in range(i+1,len(lst)):
                edges+=[(rid[lst[i]],rid[lst[j]]),(rid[lst[j]],rid[lst[i]])]
    idx=torch.tensor(edges,dtype=torch.long,device="cuda").t(); val=torch.ones(idx.shape[1],device="cuda")
    n=len(rv); A=torch.sparse_coo_tensor(idx,val,(n,n)).coalesce()
    out=torch.sparse.sum(A,1).to_dense(); out=torch.where(out==0,torch.ones_like(out),out)
    rank=torch.full((n,),1/n,device="cuda")
    for _ in range(100):
        prev=rank.clone()
        rank=0.85*torch.sparse.mm(A,(rank/out).unsqueeze(1)).squeeze()+0.15/n
        if torch.norm(rank-prev,1)<1e-6: break
    return {r:float(rank[i]) for r,i in rid.items()}

def pagerank_networkx(revs_df):
    import networkx as nx
    G=nx.Graph()
    for reviewers in revs_df.groupby("pr_id")["reviewer"].apply(set).values:
        lst=list(reviewers)
        for i in range(len(lst)):
            for j in range(i+1,len(lst)):
                G.add_edge(lst[i],lst[j])
    return nx.pagerank(G,alpha=0.85) if G.nodes else {}

# ---------------------------------------------------------------------------#
# Excel helpers                                                              #
# ---------------------------------------------------------------------------#
def sanitize_sheet(name:str)->str:
    name=re.sub(r'[\[\]\:\*\?\/\\]',"_",name)
    if len(name)>31: name=name[-31:]
    if name and name[0].isdigit(): name="_"+name
    return name or "sheet"

# ---------------------------------------------------------------------------#
# MAIN                                                                        #
# ---------------------------------------------------------------------------#
def main():
    lbl_map=load_label_map()
    initialize_feedback_table()
    update_missing_pr_labels(label_map=lbl_map)

    prs_df, files_df, revs_df = load_history()
    print("Data loaded from pr_data.db.")

    hist = build_reviewer_history(prs_df, files_df, revs_df)
    if not hist: print("No reviewer history – abort."); return

    if _HAS_CUGRAPH:
        print("Using cuGraph GPU PageRank."); pr_scores=pagerank_cugraph(revs_df)
    elif _HAS_TORCH:
        print("Using PyTorch GPU PageRank."); pr_scores=pagerank_torch(revs_df)
    else:
        print("Using networkx CPU PageRank."); pr_scores=pagerank_networkx(revs_df)
    print(f"PageRank done – reviewers ranked: {len(pr_scores)}")

    act_scores=compute_activity(revs_df)
    print("Activity scores computed.")

    while True:
        tok=input("\nEnter PR-ID for which you want recommendations (or blank to quit): ").strip()
        if not tok: return
        try:  pr_id=int(tok)
        except ValueError:
            print("PR-ID must be an integer."); continue

        pr_tags, file_paths = fetch_pr_tags_files(pr_id)
        if not file_paths:
            print(f"PR-ID {pr_id} not found or empty."); continue
        print(f"Found PR-ID {pr_id}: {len(pr_tags)} PR-level tags, {len(file_paths)} file paths.")

        dfs=[]

        for fp in file_paths:
            tags=tag_set_for_file(fp,lbl_map) or pr_tags
            tag_str=", ".join(sorted(tags)) or "–"
            print(f"\n=== Recommendations for file: {fp} | Tags: {tag_str} ===")

            ranked=rank_reviewers(tags, fp.lower(), hist,
                                  pr_scores, act_scores)

            for i,(rev,raw,nraw,act,pr,fav,nfav,fin) in enumerate(ranked[:15],1):
                print(f"{i:2d}. {rev:20s} | abs={int(raw):3d} "
                      f"norm={nraw:.3f} act={act:.3f} pr={pr:.3f} "
                      f"fav={fav:3d} nfav={nfav:.3f} final={fin:.4f}")

            df=pd.DataFrame(ranked, columns=["reviewer","raw_abs_match","normalized_abs_match",
                                             "activity_score","pagerank_score",
                                             "fav_rev_points","norm_fav_rev_points",
                                             "final_score"])
            df.insert(0,"file_path",fp)
            dfs.append((fp,df))

        try:
            from openpyxl import Workbook       # noqa: F401
            out=f"reviewer_scores_pr{pr_id}.xlsx"
            with pd.ExcelWriter(out, engine="openpyxl") as wr:
                for fp,df in dfs:
                    df.to_excel(wr,index=False,sheet_name=sanitize_sheet(fp))
            print(f"\nAll reviewer tables written to {out}")
        except Exception as e:
            print(f"[WARN] Excel export failed: {e}")

        fb=input("\nSatisfied? (y/n): ").strip().lower()
        if fb and fb[0]=="n":
            pref=input("Preferred reviewer to boost (blank to skip): ").strip()
            if pref:
                update_feedback_for_reviewer(pref,1)
                print("Feedback recorded – fav points updated.")

# ---------------------------------------------------------------------------#
if __name__=="__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
