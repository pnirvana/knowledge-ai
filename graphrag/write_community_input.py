# write_community_input.py
import argparse, pandas as pd, pathlib, sys

ROOT = pathlib.Path("graphrag_proj/output")

def _safe(s):
    s = "" if pd.isna(s) else str(s)
    # keep it simple: replace commas/newlines so we don’t break the CSV rows expected by the prompt
    return s.replace(",", " ").replace("\n", " ").replace("\r", " ").strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--community", type=int, default=None,
                    help="community id to export (level 0). If omitted, use the only community or all entities.")
    ap.add_argument("--out", default="community_input.txt", help="output text file")
    ap.add_argument("--max_entities", type=int, default=None, help="optional cap to limit the number of entities")
    ap.add_argument("--max_relationships", type=int, default=None, help="optional cap to limit the number of relationships")
    args = ap.parse_args()

    ents_pq = ROOT / "entities.parquet"
    rels_pq = ROOT / "relationships.parquet"
    comm_pq = ROOT / "communities.parquet"

    if not ents_pq.exists() or not rels_pq.exists():
        sys.exit("❌ Missing entities.parquet or relationships.parquet in graphrag_proj/output")

    entities = pd.read_parquet(ents_pq)
    relationships = pd.read_parquet(rels_pq)

    # Figure out which entities belong to the requested community (level 0)
    entity_ids = None
    if comm_pq.exists():
        communities = pd.read_parquet(comm_pq)
        if len(communities) > 0:
            # Default to first level-0 community if none specified
            if args.community is None:
                # prefer level==0 if present
                level0 = communities[communities.get("level", 0) == 0]
                row = level0.iloc[0] if len(level0) else communities.iloc[0]
            else:
                row = communities[communities["human_readable_id"] == args.community]
                if len(row) == 0:
                    sys.exit(f"❌ Community id {args.community} not found in communities.parquet")
                row = row.iloc[0]
            # entity_ids column may be list-like already; if it’s a string, try to eval‑safe split
            entity_ids = row["entity_ids"]
            if isinstance(entity_ids, str):
                # very lenient parse (handles "['a','b']" or "a,b")
                if entity_ids.startswith("[") and entity_ids.endswith("]"):
                    entity_ids = [x.strip(" '\"") for x in entity_ids[1:-1].split(",") if x.strip()]
                else:
                    entity_ids = [x.strip() for x in entity_ids.split(",") if x.strip()]

    if entity_ids:
        entities = entities[entities["id"].isin(entity_ids)]
        relationships = relationships[
            relationships["source"].isin(entities["id"]) &
            relationships["target"].isin(entities["id"])
        ]
    # else: fall back to full graph

    if args.max_entities:
        entities = entities.head(args.max_entities)
        allowed = set(entities["id"])
        relationships = relationships[
            relationships["source"].isin(allowed) &
            relationships["target"].isin(allowed)
        ]
    if args.max_relationships:
        relationships = relationships.head(args.max_relationships)

    # Build the text in the exact structure the prompt uses
    lines = []
    lines.append("Entities")
    lines.append("id,entity,description")
    for _, r in entities.iterrows():
        rid = _safe(r.get("id"))
        title = _safe(r.get("title"))
        desc = _safe(r.get("description"))
        lines.append(f"{rid},{title},{desc}")

    lines.append("")  # blank line
    lines.append("Relationships")
    lines.append("id,source,target,description")
    # relationships parquet may not have a stable id; derive one if missing
    if "id" not in relationships.columns:
        relationships = relationships.copy()
        relationships["id"] = [f"r{i}" for i in range(len(relationships))]
    for _, r in relationships.iterrows():
        rid = _safe(r.get("id"))
        src = _safe(r.get("source"))
        tgt = _safe(r.get("target"))
        desc = _safe(r.get("description"))
        lines.append(f"{rid},{src},{tgt},{desc}")

    out_path = pathlib.Path(args.out)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Wrote {len(entities)} entities and {len(relationships)} relationships to {out_path}")

if __name__ == "__main__":
    main()