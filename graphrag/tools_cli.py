import os, pathlib, json, re
import pandas as pd
import networkx as nx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from dotenv import load_dotenv
load_dotenv()

ROOT = pathlib.Path(os.getcwd())
OUT = pathlib.Path(os.getenv("GRAPH_OUTPUT_DIR", ROOT / "graphrag_proj" / "output"))

def _safe_cols(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df

def _load(name, cols=None):
    p = OUT / name
    if not p.exists():
        return pd.DataFrame(columns=cols or [])
    df = pd.read_parquet(p)
    return _safe_cols(df, cols or [])

ENT_EXPECT = ["id","title","display_name","description","type","degree","in_degree","out_degree","combined_degree","text_unit_ids"]
REL_EXPECT = ["id","source","target","edge_type","description","weight","source_short_id","target_short_id"]
COM_EXPECT = ["id","human_readable_id","community","level","entity_ids","relationship_ids","size","title"]
REP_EXPECT = ["id","human_readable_id","community","level","title","summary","full_content"]

ENTS = _load("entities.parquet", ENT_EXPECT)
RELS = _load("relationships.parquet", REL_EXPECT)
COMS = _load("communities.parquet", COM_EXPECT)
REPS = _load("community_reports.parquet", REP_EXPECT)

G = nx.DiGraph()
for _, r in ENTS.iterrows():
    G.add_node(r["id"], **r.to_dict())
for _, r in RELS.iterrows():
    G.add_edge(r["source"], r["target"], **r.to_dict())

def _series_str(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str)

_DEF_METHOD_RE = re.compile(r"^(?P<class>.+)\.(?P<method>[^.()]+)\((?P<params>.*)\)$")

def _parse_method_id(nid: str):
    m = _DEF_METHOD_RE.match(nid)
    if not m:
        return None
    cls = m.group("class")
    meth = m.group("method")
    params = [p.strip() for p in m.group("params").split(",") if p.strip()]
    params_simple = [p.split(".")[-1] for p in params]
    return cls, meth, params_simple

server = Server()

server.register_tool(Tool(name="graph.find_class", description="Find class nodes by substring.", inputSchema={"type":"object","properties":{"q":{"type":"string"},"limit":{"type":"integer"}},"required":["q"]}))
server.register_tool(Tool(name="graph.find_method", description="Find methods by class and name (optionally params).", inputSchema={"type":"object","properties":{"cls":{"type":"string"},"name":{"type":"string"},"params":{"type":"string"},"limit":{"type":"integer"}},"required":["cls","name"]}))
server.register_tool(Tool(name="graph.get_node", description="Get a node by exact id.", inputSchema={"type":"object","properties":{"id":{"type":"string"}},"required":["id"]}))
server.register_tool(Tool(name="graph.neighbors", description="List neighbors (edges) for a node.", inputSchema={"type":"object","properties":{"id":{"type":"string"},"edge_types":{"type":"array","items":{"type":"string"}},"direction":{"type":"string","enum":["out","in","both"]},"limit":{"type":"integer"}},"required":["id"]}))
server.register_tool(Tool(name="graph.ego", description="Ego subgraph nodes/edges.", inputSchema={"type":"object","properties":{"id":{"type":"string"},"depth":{"type":"integer"},"edge_types":{"type":"array","items":{"type":"string"}},"max_nodes":{"type":"integer"}},"required":["id"]}))
server.register_tool(Tool(name="graph.shortest_path", description="Shortest path between two nodes.", inputSchema={"type":"object","properties":{"src":{"type":"string"},"dst":{"type":"string"},"edge_types":{"type":"array","items":{"type":"string"}},"max_len":{"type":"integer"}},"required":["src","dst"]}))
server.register_tool(Tool(name="graph.list_communities", description="List detected communities.", inputSchema={"type":"object","properties":{"level":{"type":"integer"}}}))
server.register_tool(Tool(name="graph.get_community_report", description="Get a community report by human_readable_id.", inputSchema={"type":"object","properties":{"id":{"type":["integer","string"]}},"required":["id"]}))

@server.tool("graph.find_class")
async def t_find_class(opts: dict):
    q = (opts.get("q") or "").lower()
    limit = int(opts.get("limit") or 20)
    df = ENTS[ENTS["type"] == "Class"].copy()
    id_match = _series_str(df["id"]).str.lower().str.contains(q, regex=False)
    dn_match = _series_str(df.get("display_name", pd.Series(dtype=str))).str.lower().str.contains(q, regex=False)
    df = df[id_match | dn_match]
    rows = df.head(limit)[["id","display_name","type"]].to_dict(orient="records")
    return {"content": [TextContent(type="text", text=json.dumps(rows))]}

@server.tool("graph.find_method")
async def t_find_method(opts: dict):
    cls_token = (opts.get("cls") or "").strip()
    name = (opts.get("name") or "").strip()
    params = opts.get("params")
    limit = int(opts.get("limit") or 20)
    cls_tail = cls_token.split(".")[-1].lower()
    name_l = name.lower()
    rows = []
    for _, row in ENTS.iterrows():
        nid = str(row["id"]) ; disp = str(row.get("display_name") or "")
        parsed = _parse_method_id(nid)
        if not parsed:
            continue
        cls, meth, p_simple = parsed
        if meth.lower() != name_l:
            continue
        if cls_tail and not cls.lower().endswith(cls_tail):
            continue
        rows.append({"id": nid, "display_name": disp, "class": cls, "method": meth, "params": p_simple})
    if params:
        want = [p.strip() for p in params.split(",") if p.strip()]
        want = [w.split(".")[-1].lower() for w in want]
        rows = [r for r in rows if [p.lower() for p in r["params"]] == want]
    return {"content": [TextContent(type="text", text=json.dumps(rows[:limit]))]}

@server.tool("graph.get_node")
async def t_get_node(opts: dict):
    nid = opts.get("id")
    r = ENTS[ENTS["id"] == nid]
    found = (r.iloc[0].to_dict() if len(r) else {})
    return {"content": [TextContent(type="text", text=json.dumps(found))]}

@server.tool("graph.neighbors")
async def t_neighbors(opts: dict):
    nid = opts.get("id")
    edge_types = opts.get("edge_types") or []
    direction = opts.get("direction") or "out"
    limit = int(opts.get("limit") or 50)
    if nid not in G:
        return {"content": [TextContent(type="text", text=json.dumps([]))]}
    types = set(edge_types)
    items = []
    if direction in ("out","both"):
        for _, tgt, ed in G.out_edges(nid, data=True):
            if types and ed.get("edge_type") not in types: continue
            items.append({"source": nid, "target": tgt, "edge_type": ed.get("edge_type"), "description": ed.get("description")})
    if direction in ("in","both"):
        for src, _, ed in G.in_edges(nid, data=True):
            if types and ed.get("edge_type") not in types: continue
            items.append({"source": src, "target": nid, "edge_type": ed.get("edge_type"), "description": ed.get("description")})
    return {"content": [TextContent(type="text", text=json.dumps(items[:limit]))]}

@server.tool("graph.ego")
async def t_ego(opts: dict):
    nid = opts.get("id")
    depth = int(opts.get("depth") or 2)
    edge_types = opts.get("edge_types") or []
    max_nodes = int(opts.get("max_nodes") or 200)
    if nid not in G:
        return {"content": [TextContent(type="text", text=json.dumps({"nodes":[],"edges":[]}))]}
    H = nx.DiGraph((u,v,d) for u,v,d in G.edges(data=True) if not edge_types or d.get("edge_type") in set(edge_types))
    seen = {nid}; frontier = [nid]
    for _ in range(depth):
        nxt = []
        for u in frontier:
            for _, v in H.out_edges(u):
                if len(seen) >= max_nodes: break
                if v not in seen: seen.add(v); nxt.append(v)
            for v, _ in H.in_edges(u):
                if len(seen) >= max_nodes: break
                if v not in seen: seen.add(v); nxt.append(v)
        frontier = nxt
        if not frontier or len(seen) >= max_nodes: break
    nodes = [{"id": n, "display_name": (G.nodes[n].get("display_name") if n in G.nodes else n), "type": (G.nodes[n].get("type") if n in G.nodes else None)} for n in seen]
    edges = [{"source": u, "target": v, "edge_type": d.get("edge_type"), "description": d.get("description")} for u,v,d in H.subgraph(seen).edges(data=True)]
    return {"content": [TextContent(type="text", text=json.dumps({"nodes":nodes, "edges":edges}))]}

@server.tool("graph.shortest_path")
async def t_shortest(opts: dict):
    src = opts.get("src") ; dst = opts.get("dst")
    edge_types = opts.get("edge_types") or []
    max_len = int(opts.get("max_len") or 12)
    H = nx.DiGraph((u,v,d) for u,v,d in G.edges(data=True) if not edge_types or d.get("edge_type") in set(edge_types))
    try:
        path = nx.shortest_path(H, src, dst)
        if len(path) > max_len + 1:
            return {"content": [TextContent(type="text", text=json.dumps({"error":"path too long","len":len(path)}))]}
        return {"content": [TextContent(type="text", text=json.dumps({"path":path}))]}
    except Exception:
        return {"content": [TextContent(type="text", text=json.dumps({"path":[]}))]}

@server.tool("graph.list_communities")
async def t_list_comms(opts: dict):
    level = opts.get("level")
    df = COMS
    if level is not None:
        try:
            df = df[df.get("level",0) == int(level)]
        except Exception:
            pass
    cols = [c for c in ["human_readable_id","community","level","size","title"] if c in df.columns]
    return {"content": [TextContent(type="text", text=json.dumps(df[cols].to_dict(orient="records")))]}

@server.tool("graph.get_community_report")
async def t_get_comm_report(opts: dict):
    id_ = opts.get("id")
    try:
        id_int = int(id_)
        row = REPS[REPS["human_readable_id"] == id_int]
    except Exception:
        row = REPS[REPS["human_readable_id"] == id_]
    if len(row)==0:
        return {"content": [TextContent(type="text", text=json.dumps({}))]}
    row = row.iloc[0]
    out = {"title":row.get("title"), "summary":row.get("summary"), "full_content":row.get("full_content")}
    return {"content": [TextContent(type="text", text=json.dumps(out))]}

import asyncio

async def _main():
    async with stdio_server(server) as (read, write):
        await server.run(read, write, initialization_options={})

if __name__ == "__main__":
    asyncio.run(_main())