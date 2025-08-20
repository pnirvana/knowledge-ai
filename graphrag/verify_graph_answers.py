#!/usr/bin/env python3
import re
from pathlib import Path
import pandas as pd
import networkx as nx

ROOT = Path("graphrag_proj/output")
E = pd.read_parquet(ROOT / "entities.parquet")
R = pd.read_parquet(ROOT / "relationships.parquet")

def search_entities(pattern, field="title"):
    rx = re.compile(pattern, re.I)
    cols = ["id","title","type","combined_degree"]
    return E[E[field].fillna("").str.contains(rx)].sort_values("combined_degree", ascending=False)[cols]

def publishers_and_events():
    rels = R[R["edge_type"]=="publishes_event"][["source","target"]]
    src = E.set_index("id")[["title","type"]]
    tgt = E.set_index("id")[["title","type"]]
    out = rels.join(src, on="source", rsuffix="_src").join(tgt, on="target", rsuffix="_tgt")
    out = out.rename(columns={"title":"publisher","type":"publisher_type","title_tgt":"event","type_tgt":"event_type"})
    return out[["source","publisher","publisher_type","target","event","event_type"]].drop_duplicates()

def calls_of(method_pattern):
    rx = re.compile(method_pattern, re.I)
    # find matching callee nodes
    callees = E[E["title"].fillna("").str.contains(rx)]["id"].tolist()
    rels = R[(R["edge_type"]=="calls") & (R["target"].isin(callees))][["source","target"]]
    titles = E.set_index("id")["title"]
    rels["source_title"] = rels["source"].map(titles)
    rels["target_title"] = rels["target"].map(titles)
    return rels.drop_duplicates().sort_values(["source_title","target_title"])

def neighbors(node_pattern, edge_type=None, direction="out"):
    rx = re.compile(node_pattern, re.I)
    nodes = E[E["title"].fillna("").str.contains(rx)]["id"].tolist()
    if not nodes:
        return pd.DataFrame()
    rels = R
    if edge_type:
        rels = rels[rels["edge_type"]==edge_type]
    if direction=="out":
        rels = rels[rels["source"].isin(nodes)]
        col = "target"
    else:
        rels = rels[rels["target"].isin(nodes)]
        col = "source"
    titles = E.set_index("id")[["title","type"]]
    rels = rels.join(titles, on=col, rsuffix="_nbr")
    return rels[[col,"title","type","edge_type"]].drop_duplicates().rename(columns={col:"id","title":"neighbor","type":"neighbor_type"})

def shortest_path(src_pattern, dst_pattern, max_hops=6):
    G = nx.from_pandas_edgelist(R, source="source", target="target", edge_attr="edge_type", create_using=nx.DiGraph())
    rx_s = re.compile(src_pattern, re.I)
    rx_t = re.compile(dst_pattern, re.I)
    src_ids = E[E["title"].fillna("").str.contains(rx_s)]["id"].tolist()
    dst_ids = E[E["title"].fillna("").str.contains(rx_t)]["id"].tolist()
    titles = E.set_index("id")["title"].to_dict()
    for s in src_ids:
        for t in dst_ids:
            try:
                path = nx.shortest_path(G, s, t)
                if len(path)-1 <= max_hops:
                    return [{"id":nid, "title":titles.get(nid, nid)} for nid in path]
            except nx.NetworkXNoPath:
                continue
    return []

if __name__ == "__main__":
    print("# Publishers and Events (ground truth)")
    print(publishers_and_events().head(20).to_string(index=False))

    print("\n# Who calls GetOrderResponse?")
    print(calls_of(r"GetOrderResponse").head(20).to_string(index=False))

    print("\n# Neighbors of OrderController (outgoing calls)")
    print(neighbors(r"OrderController", edge_type="calls", direction="out").head(20).to_string(index=False))

    print("\n# Shortest path OrderController -> OrderCancelled (any edge)")
    print(shortest_path(r"OrderController", r"OrderCancelled", max_hops=6))