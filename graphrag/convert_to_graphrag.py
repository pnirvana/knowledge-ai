#!/usr/bin/env python3
# convert_to_graphrag.py
import json
import uuid
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd


# -------------------
# Config
# -------------------
NODES_JSON = Path("graph_nodes.json")
EDGES_JSON = Path("graph_edges.json")
PROJECT_ROOT = Path("graphrag_proj")
OUT_DIR = PROJECT_ROOT / "output"     # GraphRAG BYOG expects Parquet in output/
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAKE_FALLBACK_COMMUNITIES = False  # set False if you want GraphRAG to compute them


# -------------------
# Load JSON
# -------------------
nodes = json.loads(NODES_JSON.read_text(encoding="utf-8"))
edges = json.loads(EDGES_JSON.read_text(encoding="utf-8"))

# Basic DataFrames
nodes_df = pd.DataFrame(nodes)
edges_df = pd.DataFrame(edges)

# Ensure required columns exist to avoid KeyErrors later
for col in ["id", "name", "type", "description", "file", "return_type", "parameters", "attr_type"]:
    if col not in nodes_df.columns:
        nodes_df[col] = None

for col in ["from", "to", "type"]:
    if col not in edges_df.columns:
        edges_df[col] = None

# -------------------
# Compute degrees from edges
# -------------------
deg_in = Counter()
deg_out = Counter()
deg_tot = Counter()

for s, t in edges_df[["from", "to"]].itertuples(index=False, name=None):
    if s is None or t is None:
        continue
    deg_out[s] += 1
    deg_in[t] += 1
    deg_tot[s] += 1
    deg_tot[t] += 1

# -------------------
# Build Entities
# -------------------
def compact_description(nrow: pd.Series) -> str:
    bits = []
    if nrow.get("type"):
        bits.append(f"type={nrow['type']}")
    if nrow.get("return_type"):
        bits.append(f"returns={nrow['return_type']}")
    params_val = nrow.get("parameters")
    if isinstance(params_val, (list, tuple)) and params_val:
        bits.append("params=" + ", ".join(map(str, params_val)))
    if nrow.get("attr_type"):
        bits.append(f"attr={nrow['attr_type']}")
    if nrow.get("file"):
        bits.append(f"file={nrow['file']}")
    if nrow.get("description"):
        bits.append(str(nrow["description"]).strip())
    return " | ".join(bits) if bits else (nrow.get("name") or nrow["id"])

entities = pd.DataFrame({
    "id": nodes_df["id"],
    # IMPORTANT: GraphRAG community builder expects title to align with ids used in relationships
    "title": nodes_df["id"],
    "description": nodes_df.apply(compact_description, axis=1),
    "type": nodes_df["type"].fillna(""),
})

# Degrees
entities["in_degree"] = entities["id"].map(lambda x: int(deg_in.get(x, 0)))
entities["out_degree"] = entities["id"].map(lambda x: int(deg_out.get(x, 0)))
entities["combined_degree"] = entities["id"].map(lambda x: int(deg_tot.get(x, 0)))
entities["degree"] = entities["combined_degree"]

# Human-friendly label for UIs and text units
_short_display = nodes_df.apply(lambda r: (r.get("name") or str(r.get("id", "")).split(".")[-1]), axis=1)
entities["display_name"] = _short_display.astype(str)

# Display name & text units (linked later)
entities["text_unit_ids"] = [[] for _ in range(len(entities))]

# Stable numeric short ids (GraphRAG expects hrid to be int)
entities = entities.sort_values(["display_name", "id"], kind="stable").reset_index(drop=True)
entities["human_readable_id"] = entities.index.astype("int64")
entities["short_id"] = entities["human_readable_id"]

# Maps
id2short = dict(zip(entities["id"], entities["short_id"]))
id2cdeg = dict(zip(entities["id"], entities["combined_degree"]))
entity_titles = dict(zip(entities["id"], entities["display_name"]))

# -------------------
# Build Relationships
# -------------------
DEFAULT_W = {
    "calls": 1.0,
    "implemented_by": 1.0,
    "has_attribute": 0.6,
    "reads_attribute": 0.9,
    "writes_attribute": 0.9,
    "creates": 0.7,
    "publishes_event": 1.2,
    "defines": 0.6,
}

rels = pd.DataFrame({
    "id": [str(uuid.uuid4()) for _ in range(len(edges_df))],
    "source": edges_df["from"],
    "target": edges_df["to"],
    "edge_type": edges_df["type"].fillna("related_to").astype(str),
    "description": edges_df["type"].fillna("related_to").astype(str),
    "weight": edges_df["type"].map(DEFAULT_W).fillna(1.0).astype(float),
})
rels["text_unit_ids"] = [[] for _ in range(len(rels))]

# Degree features required by GraphRAG reports
rels["source_combined_degree"] = rels["source"].map(id2cdeg).fillna(0).astype("int64")
rels["target_combined_degree"] = rels["target"].map(id2cdeg).fillna(0).astype("int64")
rels["combined_degree"] = (rels["source_combined_degree"] + rels["target_combined_degree"]).astype("int64")

# Stable numeric hrid for edges
rels = rels.sort_values(["edge_type", "source", "target", "id"], kind="stable").reset_index(drop=True)
rels["human_readable_id"] = rels.index.astype("int64")

# Wire source/target short ids (handy for joins)
rels["source_short_id"] = rels["source"].map(id2short).fillna(-1).astype("int64")
rels["target_short_id"] = rels["target"].map(id2short).fillna(-1).astype("int64")

# -------------------
# (Optional) Fallback communities
# -------------------
if MAKE_FALLBACK_COMMUNITIES:
    # Single community containing everything; GraphRAG can still summarize
    comm = pd.DataFrame([{
        "id": "C0",
        "human_readable_id": 0,     # int
        "community": 0,             # int
        "level": 0,
        "parent": None,
        "children": [],
        "title": "All Components",
        "entity_ids": entities["id"].tolist(),
        "relationship_ids": rels["id"].tolist(),
        "text_unit_ids": [],
        "period": None,
        "size": int(len(entities)),
    }])
    comm.to_parquet(OUT_DIR / "communities.parquet", index=False)

# -------------------
# Build Text Units (richer content for Global)
# -------------------
node_meta = nodes_df.set_index("id")[["name", "file", "return_type", "parameters", "type"]].to_dict(orient="index")

# Build neighbor summaries per source and edge_type
neighbors = defaultdict(lambda: defaultdict(list))
for _, erow in rels.iterrows():
    s = erow["source"]; t = erow["target"]; et = erow["edge_type"]
    if s is None or t is None:
        continue
    neighbors[s][et].append(t)

# Build publish pairs for the global domain summary
publish_rows = rels[rels["edge_type"] == "publishes_event"][["source", "target"]].dropna()
publish_pairs = []
for _, pr in publish_rows.iterrows():
    s = pr["source"]; t = pr["target"]
    if not s or not t:
        continue
    s_title = entity_titles.get(s, s.split(".")[-1])
    t_title = entity_titles.get(t, t.split(".")[-1])
    publish_pairs.append((s, t, s_title, t_title))

# Map method -> enclosing class and accumulate per-class publish lines
entity_id_set = set(entities["id"])
from collections import defaultdict
class_pub_lines = defaultdict(list)
for s_id, t_id, s_title, t_title in publish_pairs:
    # Derive class id by trimming after the last '.'
    cls_id = s_id.rsplit(".", 1)[0] if "." in s_id else None
    if cls_id and cls_id in entity_id_set:
        class_pub_lines[cls_id].append(f"{s_title} publishes event {t_title}.")

def _unique_titles(ids, limit=8):
    seen = set()
    out = []
    for nid in ids:
        if nid in seen:
            continue
        seen.add(nid)
        out.append(entity_titles.get(nid, nid.split(".")[-1]))
        if len(out) >= limit:
            break
    return out

def _signature(nid):
    m = node_meta.get(nid, {}) or {}
    name = m.get("name") or entity_titles.get(nid, nid.split(".")[-1])
    rtype = m.get("return_type") or ""
    params = m.get("parameters")
    # robust params -> text
    ptxt = ""
    try:
        if params is None:
            ptxt = ""
        elif isinstance(params, float):  # NaN
            ptxt = ""
        elif isinstance(params, str):
            ptxt = params
        elif isinstance(params, (list, tuple, set)):
            ptxt = ", ".join(map(str, params))
        else:
            try:
                ptxt = ", ".join(map(str, list(params)))
            except Exception:
                ptxt = str(params)
    except Exception:
        ptxt = ""
    rtype_str = "" if (rtype is None or isinstance(rtype, float)) else str(rtype)
    name_str  = "" if (name  is None or isinstance(name,  float)) else str(name)
    return f"{rtype_str + ' ' if rtype_str else ''}{name_str}({ptxt})"

def _file(nid):
    return (node_meta.get(nid, {}) or {}).get("file")

def _etype(nid):
    return (node_meta.get(nid, {}) or {}).get("type") or ""

text_units_rows = []

# Entity text units: include type, file, signature, neighbor rollups, and explicit publish lines
for _, row in entities.iterrows():
    nid = row["id"]
    title = row.get("display_name", row["title"])
    etype = row.get("type") or _etype(nid) or ""
    file_path = _file(nid)
    sig = _signature(nid)

    # Summaries
    calls_list = _unique_titles(neighbors[nid].get("calls", []), limit=8)
    pub_list   = _unique_titles(neighbors[nid].get("publishes_event", []), limit=8)
    attr_list  = _unique_titles(neighbors[nid].get("has_attribute", []), limit=8)

    # Explicit declarative sentences for Global summarizer
    explicit_pub_lines = [f"{title} publishes event {ev}." for ev in pub_list]

    lines = [f"{title} — {etype}".strip()]
    if file_path:
        lines.append(f"file: {file_path}")
    if sig:
        lines.append(f"signature: {sig}")

    if calls_list:
        lines.append("calls → [" + ", ".join(calls_list) + "]")
    if pub_list:
        lines.append("publishes_event → [" + ", ".join(pub_list) + "]")
    if attr_list:
        lines.append("has_attribute → [" + ", ".join(attr_list) + "]")

    # Add explicit publish sentences (helps map/reduce latch onto evidence)
    if explicit_pub_lines:
        lines.extend(explicit_pub_lines)

    # If this is a class and its methods publish events, add those lines too
    if nid in class_pub_lines:
        lines.extend(class_pub_lines[nid])

    # fallback to prior compact description if everything above is empty
    if len(lines) <= 1 and isinstance(row.get("description"), str) and row["description"].strip():
        lines.append(str(row["description"]).strip())

    text_units_rows.append({
        "id": f"tu_e_{row['human_readable_id']}",
        "text": "\n".join(lines),
        "type": "entity_text",
        "source_id": nid,
        "source_type": "entity",
    })

# Relationship text units: readable triple with titles and weight
for _, row in rels.iterrows():
    s = row["source"]; t = row["target"]; et = row["edge_type"]
    st = entity_titles.get(s, s.split(".")[-1])
    tt = entity_titles.get(t, t.split(".")[-1])
    wt = row.get("weight", 1.0)
    text = f"{st} —({et}, weight={wt})-> {tt}"
    text_units_rows.append({
        "id": f"tu_r_{row['human_readable_id']}",
        "text": text,
        "type": "relationship_text",
        "source_id": row["id"],
        "source_type": "relationship",
    })

# --- Domain Summary: explicit sentences for Global to latch onto
if publish_pairs:
    summary_lines = ["Domain event publishing summary:"]
    for _, _, s_title, t_title in publish_pairs:
        summary_lines.append(f"{s_title} publishes event {t_title}.")
    summary_text = "\n".join(summary_lines)

    # Append a new summary entity
    next_hrid = int(entities["human_readable_id"].max()) + 1 if len(entities) else 0
    summary_entity_id = "domain_summary_0"
    summary_row = {
        "id": summary_entity_id,
        "title": "Domain Event Publishing Summary",
        "description": "Auto-generated summary of publish relationships.",
        "type": "Summary",
        "in_degree": 0,
        "out_degree": 0,
        "combined_degree": 0,
        "degree": 0,
        "display_name": "Domain Event Publishing Summary",
        "text_unit_ids": [],
        "human_readable_id": next_hrid,
        "short_id": next_hrid,
    }
    entities = pd.concat([entities, pd.DataFrame([summary_row])], ignore_index=True)
    entity_titles[summary_entity_id] = "Domain Event Publishing Summary"

    # Append its text unit
    text_units_rows.append({
        "id": "tu_summary_0",
        "text": summary_text,
        "type": "entity_text",
        "source_id": summary_entity_id,
        "source_type": "entity",
    })

text_units = pd.DataFrame(text_units_rows)

# Link text units back to entities/relationships so GraphRAG can assemble context
entities["text_unit_ids"] = entities["human_readable_id"].apply(lambda x: [f"tu_e_{x}"] if f"tu_e_{x}" in set(text_units["id"]) else [])
rels["text_unit_ids"] = rels["human_readable_id"].apply(lambda x: [f"tu_r_{x}"] if f"tu_r_{x}" in set(text_units["id"]) else [])

# Ensure the summary entity points to its TU
if "domain_summary_0" in entities["id"].values:
    entities.loc[entities["id"] == "domain_summary_0", "text_unit_ids"] = entities.loc[
        entities["id"] == "domain_summary_0", "text_unit_ids"
    ].apply(lambda _: ["tu_summary_0"])

# -------------------
# Write Parquet
# -------------------
entities.to_parquet(OUT_DIR / "entities.parquet", index=False)
rels.to_parquet(OUT_DIR / "relationships.parquet", index=False)
text_units.to_parquet(OUT_DIR / "text_units.parquet", index=False)

print(f"✅ Wrote entities={len(entities)} relationships={len(rels)} text_units={len(text_units)} to {OUT_DIR}")
print("   Columns (entities):", list(entities.columns))
print("   Columns (relationships):", list(rels.columns))
print("   Columns (text_units):", list(text_units.columns))
# Refresh fallback community to include any new entities added after it was written
if MAKE_FALLBACK_COMMUNITIES:
    cpath = OUT_DIR / "communities.parquet"
    try:
        c = pd.read_parquet(cpath)
        if len(c) > 0:
            c.at[0, "entity_ids"] = entities["id"].tolist()
            c.to_parquet(cpath, index=False)
    except Exception as e:
        print("Warning: could not refresh communities.parquet:", e)