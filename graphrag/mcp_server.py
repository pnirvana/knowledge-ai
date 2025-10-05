#!/usr/bin/env python3
"""
Code-Graph MCP server
- Exposes your entities/relationships/communities/community_reports over MCP tools
- Optional source snippets (node_sources.parquet)
- Optional FAISS semantic search over text_units (if present)

Env:
  GRAPH_OUTPUT_DIR   : path to graphrag output dir (default: ./graphrag_proj/output)
  CODEGRAPH_ALLOWLIST: colon-separated allowlisted source roots for open_file/read_source
"""

import os, json, pathlib, re
import asyncio
from typing import List, Optional, Dict, Any
import uvicorn
import asyncio

# --- JSON sanitization helpers ---
def _json_sanitize(obj):
    try:
        import numpy as _np  # type: ignore
    except Exception:
        _np = None

    if obj is None:
        return None
    # Simple primitives
    if isinstance(obj, (str, int, float, bool)):
        # Normalize NaN/inf
        if isinstance(obj, float):
            if obj != obj:  # NaN
                return None
            if obj == float("inf") or obj == float("-inf"):
                return None
        return obj
    # Path-like
    if isinstance(obj, pathlib.Path):
        return str(obj)
    # Mapping
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    # Set/Tuple/List-like
    if isinstance(obj, (set, tuple, list)):
        return [_json_sanitize(v) for v in obj]
    # Pandas scalar
    try:
        import pandas as _pd  # noqa
        from pandas.api.types import is_scalar as _pd_is_scalar
        if _pd_is_scalar(obj):
            return _json_sanitize(obj.item() if hasattr(obj, "item") else obj)
    except Exception:
        pass
    # NumPy arrays/scalars
    if _np is not None:
        if hasattr(obj, "dtype") and hasattr(obj, "shape"):
            try:
                return [_json_sanitize(v) for v in obj.tolist()]
            except Exception:
                return str(obj)
        if hasattr(_np, "generic") and isinstance(obj, _np.generic):
            try:
                return _json_sanitize(obj.item())
            except Exception:
                return str(obj)
    # Fallback to string
    try:
        return json.loads(obj) if isinstance(obj, str) and obj.strip().startswith("{") else str(obj)
    except Exception:
        return str(obj)

import pandas as pd
import networkx as nx

# MCP SDK
import sys
from mcp.server.fastmcp import FastMCP
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# ---------- Config ----------
ROOT = pathlib.Path(os.getcwd())
OUT = pathlib.Path(os.getenv("GRAPH_OUTPUT_DIR", ROOT / "graphrag_proj" / "output"))

# Allowlist for reading source files
def _parse_allowlist() -> List[pathlib.Path]:
    raw = os.getenv("CODEGRAPH_ALLOWLIST")
    if raw:
        return [pathlib.Path(p.strip()) for p in raw.split(":") if p.strip()]
    # Fallback: only current repo root
    return [ROOT]

ALLOW_SRCS = _parse_allowlist()

# ---------- Utilities ----------
def _safe_cols(df: pd.DataFrame, expected: List[str]) -> pd.DataFrame:
    missing = [c for c in expected if c not in df.columns]
    for c in missing:
        df[c] = None
    return df

def _load_pq(name: str, expected: Optional[List[str]] = None) -> pd.DataFrame:
    p = OUT / name
    if not p.exists():
        return pd.DataFrame(columns=expected or [])
    df = pd.read_parquet(p)
    if expected:
        df = _safe_cols(df, expected)
    return df

def _in_allowlist(p: pathlib.Path) -> bool:
    try:
        rp = p.resolve()
    except Exception:
        return False
    for root in ALLOW_SRCS:
        try:
            if str(rp).startswith(str(root.resolve())):
                return True
        except Exception:
            pass
    return False

def _read_slice(path: str, start: int | None, end: int | None, surround: int = 0):
    p = pathlib.Path(path)
    if p.exists() and p.is_dir():
        return None, "path is a directory"
    if not _in_allowlist(p):
        return None, "path not allowlisted"
    try:
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception as e:
        return None, f"read error: {e}"

    # Normalize start/end strictly; no surround here.
    s = int(start) if isinstance(start, (int,)) and start and start > 0 else 1
    e = int(end) if isinstance(end, (int,)) and end and (end >= s) else s

    s = max(1, s)
    e = min(len(lines), e)

    buf = []
    for i in range(s, e + 1):
        if 1 <= i <= len(lines):
            buf.append(f"{i:>6}: {lines[i-1]}")
    return "\n".join(buf), None


# --- new helpers ---
from typing import Optional

def _coalesce_int(*vals, default: Optional[int] = None) -> Optional[int]:
    """Return the first usable integer from vals; skips None/NaN/empty strings. Returns `default` (possibly None) if none found."""
    for v in vals:
        if v is None:
            continue
        # Skip NaN
        try:
            import math
            if isinstance(v, float) and math.isnan(v):
                continue
        except Exception:
            pass
        if isinstance(v, int):
            return int(v)
        if isinstance(v, str):
            s = v.strip()
            if s.isdigit():
                return int(s)
    return default


def _pretty_method_signature(nid: str) -> Optional[str]:
    """Return a compact Class.method(ShortParam,...) signature if nid looks like a method id."""
    m = re.match(r"^(?P<cls>.+)\.(?P<meth>[^.()]+)\((?P<params>.*)\)$", nid or "")
    if not m:
        return None
    cls_tail = m.group("cls").split(".")[-1]
    meth = m.group("meth")
    params = m.group("params")
    short_params = ",".join([p.strip().split(".")[-1] for p in params.split(",") if p.strip()])
    return f"{cls_tail}.{meth}({short_params})"


def _as_list(x) -> List[Any]:
    """Normalize various container-ish values into a plain Python list."""
    if x is None:
        return []
    # Already a list
    if isinstance(x, list):
        return x
    # Pandas Series
    try:
        import pandas as _pd  # type: ignore
        if isinstance(x, _pd.Series):
            return [v for v in x.tolist() if v is not None]
    except Exception:
        pass
    # NumPy array
    try:
        import numpy as _np  # type: ignore
        if isinstance(x, _np.ndarray):  # type: ignore
            return [v for v in x.tolist() if v is not None]
    except Exception:
        pass
    # Stringified list / CSV
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].strip()
            if not inner:
                return []
            parts = [p.strip().strip("'").strip('"') for p in inner.split(",")]
            return [p for p in parts if p]
        return [t.strip() for t in s.split(",") if t.strip()]
    # Single scalar
    return [x]

# ---------- Load data ----------
ENT_EXPECT = ["id","title","display_name","description","type","degree","in_degree","out_degree","combined_degree","text_unit_ids"]
REL_EXPECT = ["id","source","target","edge_type","description","weight","source_short_id","target_short_id"]
COM_EXPECT = ["id","human_readable_id","community","level","entity_ids","relationship_ids","size","title"]
REP_EXPECT = ["id","human_readable_id","community","level","title","summary","full_content"]

ENTS = _load_pq("entities.parquet", ENT_EXPECT)
RELS = _load_pq("relationships.parquet", REL_EXPECT)
COMS = _load_pq("communities.parquet", COM_EXPECT)
REPS = _load_pq("community_reports.parquet", REP_EXPECT)
TXT  = _load_pq("text_units.parquet", ["id","text","type","source_id","source_type"])

NODE_SRC = None
for cand in ["node_sources.parquet", "nodes_sources.parquet"]:
    p = OUT / cand
    if p.exists():
        NODE_SRC = pd.read_parquet(p)
        NODE_SRC = _safe_cols(NODE_SRC, ["id","file","path","start_line","end_line","signature"])
        break

# Build graph (directed)
G = nx.DiGraph()
for _, r in ENTS.iterrows():
    G.add_node(r["id"], **r.to_dict())
for _, r in RELS.iterrows():
    G.add_edge(r["source"], r["target"], **r.to_dict())

# ---------- Optional FAISS semantic search ----------
_FAISS_OK = False
_FAISS = None
_FAISS_INDEX = None
_FAISS_META = None

def _init_faiss():
    global _FAISS_OK, _FAISS, _FAISS_INDEX, _FAISS_META
    try:
        import faiss  # type: ignore
        _FAISS = faiss
    except Exception:
        _FAISS_OK = False
        return
    idx_path = OUT / "text_units.faiss"
    meta_path = OUT / "text_units_meta.parquet"
    if not idx_path.exists() or not meta_path.exists():
        _FAISS_OK = False
        return
    try:
        _FAISS_INDEX = _FAISS.read_index(str(idx_path))
        _FAISS_META = pd.read_parquet(meta_path)
        _FAISS_OK = True
    except Exception:
        _FAISS_OK = False

_init_faiss()

def _semantic_search(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    if not _FAISS_OK:
        return []
    # Embed with a cheap local model? Here we assume you pre-embedded; this call
    # is a no-op without an embedding function. For now, return empty to avoid surprises.
    # (You can swap in your embedding function and search vectors here.)
    return []

# ---------- Helper lookups ----------
def _node_meta(nid: str) -> Optional[Dict[str, Any]]:
    rows = ENTS[ENTS["id"] == nid]
    return rows.iloc[0].to_dict() if len(rows) else None

def _neighbors(nid: str, edge_types: Optional[List[str]], direction: str, limit: int) -> List[Dict[str, Any]]:
    if nid not in G:
        return []
    items = []
    if direction in ("out", "both"):
        for _, tgt, ed in G.out_edges(nid, data=True):
            if edge_types and ed.get("edge_type") not in edge_types:
                continue
            items.append({"source": nid, "target": tgt, "edge_type": ed.get("edge_type"), "description": ed.get("description")})
    if direction in ("in", "both"):
        for src, _, ed in G.in_edges(nid, data=True):
            if edge_types and ed.get("edge_type") not in edge_types:
                continue
            items.append({"source": src, "target": nid, "edge_type": ed.get("edge_type"), "description": ed.get("description")})
    return items[:limit]


# --- Merge line-hints from all sources ---
def _collect_hints(*pairs: tuple[Optional[int], Optional[int]]):
    """Merge multiple (start,end) pairs, returning (min_start, max_end) if any usable values exist."""
    starts, ends = [], []
    for a, b in pairs:
        try:
            if a is not None:
                a = int(a)
                if a > 0:
                    starts.append(a)
        except Exception:
            pass
        try:
            if b is not None:
                b = int(b)
                if b > 0:
                    ends.append(b)
        except Exception:
            pass
    if not starts and not ends:
        return None, None
    s = min(starts) if starts else None
    e = max(ends) if ends else None
    if s is not None and e is not None and e < s:
        e = s
    return s, e

def _file_len(p: pathlib.Path) -> int:
    try:
        if p.exists() and p.is_file():
            return len(p.read_text(encoding="utf-8", errors="ignore").splitlines())
    except Exception:
        pass
    return 0

def _resolve_source(nid: str) -> Optional[Dict[str, Any]]:
    """
    Resolve a node to a concrete source path and optional line-hints.

    Rules (per user request):
      - Do **not** try to re-root or search for alternate copies of the file.
      - Use the path exactly as given in node_sources (preferred) or entities/description.
      - If the file doesn't exist **as-is**, return None (the caller will emit a clear error).
      - Only return hints if they are present; otherwise leave them unset so the caller may guess.
    """
    import re, pathlib

    def _parse_hint_blob(s: str) -> tuple[str, Optional[int], Optional[int]]:
        """Split `"/path/Foo.java (L8-102)"` into (path, 8, 102). If no hint, returns (path, None, None)."""
        s = (s or "").strip()
        m = re.match(r"^(?P<path>.*?)(?:\s*\(L\s*(?P<s>\d+)\s*-\s*(?P<e>\d+)\))?$", s)
        if not m:
            return s, None, None
        p = m.group("path").strip()
        s_hint = int(m.group("s")) if m.group("s") else None
        e_hint = int(m.group("e")) if m.group("e") else None
        return p, s_hint, e_hint

    def _path_from_desc(desc: str) -> tuple[Optional[str], Optional[int], Optional[int]]:
        if not isinstance(desc, str) or not desc:
            return None, None, None
        m = re.search(r"file\s*=\s*([^|\n]+)", desc, flags=re.I)
        if not m:
            return None, None, None
        raw = m.group(1).strip()
        p, s_hint, e_hint = _parse_hint_blob(raw)
        return p, s_hint, e_hint

    # 1) Prefer node_sources mapping
    raw_path: Optional[str] = None
    s_hint: Optional[int] = None
    e_hint: Optional[int] = None
    signature: Optional[str] = None

    if NODE_SRC is not None:
        rows = NODE_SRC[NODE_SRC["id"] == nid]
        if len(rows):
            r = rows.iloc[0].to_dict()
            raw_path = r.get("file") or r.get("path") or None
            signature = r.get("signature") or r.get("display_name") or r.get("title")
            if isinstance(raw_path, str):
                raw_path, s0, e0 = _parse_hint_blob(raw_path)
                s_hint = s0 if s0 is not None else r.get("start_line")
                e_hint = e0 if e0 is not None else r.get("end_line")
            else:
                s_hint = r.get("start_line"); e_hint = r.get("end_line")

    # 2) Fall back to entity row (file/path columns or description)
    if raw_path is None:
        mrow = _node_meta(nid)
        if mrow:
            col_path = mrow.get("file") or mrow.get("path") or None
            if isinstance(col_path, str) and col_path.strip():
                col_path, s1, e1 = _parse_hint_blob(col_path)
            else:
                col_path, s1, e1 = _path_from_desc(mrow.get("description"))
            raw_path = col_path
            # Prefer explicit column hints if present, else hint from description
            s_hint = _coalesce_int(mrow.get("start_line"), s_hint, s1)
            e_hint = _coalesce_int(mrow.get("end_line"), e_hint, e1)
            if signature is None:
                signature = mrow.get("display_name") or mrow.get("title")

    if not raw_path or not str(raw_path).strip():
        return None

    # 3) Use the path exactly as-is (no re-rooting)
    p = pathlib.Path(str(raw_path).strip())
    if not p.exists() or not p.is_file():
        return None
    if not _in_allowlist(p):
        # Path exists but is not allowlisted; treat as unresolved to force a clear error upstream
        return None

    # 4) Normalize/cap hints to file length if possible
    file_len = _file_len(p)
    try:
        if s_hint is not None:
            s_hint = max(1, int(s_hint))
        if e_hint is not None:
            e_hint = max(1, int(e_hint))
            if s_hint is not None and e_hint < s_hint:
                e_hint = s_hint
        if file_len and e_hint is not None:
            e_hint = min(int(e_hint), file_len)
    except Exception:
        pass

    out: Dict[str, Any] = {"path": str(p), "signature": signature, "file_length": file_len}
    if s_hint is not None:
        out["start_line"] = int(s_hint)
    if e_hint is not None:
        out["end_line"] = int(e_hint)
    return out


app = FastAPI()
mcp = FastMCP("codegraph-mcp")

# --- Heuristic span guessing for source fallback ---
def _guess_span(path: str, nid: str) -> tuple[Optional[int], Optional[int], Optional[str]]:
    """
    Best-effort: derive (start_line, end_line) for a node by scanning the file.
    - Works for Java-like syntax with braces.
    - If nid looks like Class.method(params), we search for that method.
    - Otherwise we try to find `class ClassName` and return the class block.
    Returns (start, end, signature) or (None, None, None) on failure.
    """
    p = pathlib.Path(path)
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None, None, None

    lines = text.splitlines()
    # Parse nid
    m = re.match(r"^(?P<cls>.+?)\.(?P<meth>[^.()]+)\((?P<params>.*)\)$", nid or "")
    cls_tail = None
    meth = None
    short_params: list[str] = []
    if m:
        cls_tail = m.group("cls").split(".")[-1]
        meth = m.group("meth")
        params = [p.strip() for p in m.group("params").split(",") if p.strip()]
        short_params = [p.split(".")[-1] for p in params]
    else:
        # Try class-only id
        m2 = re.match(r"^(?P<cls>.+?)$", nid or "")
        if m2:
            cls_tail = m2.group("cls").split(".")[-1]

    def _find_block_start(patterns: list[str]) -> Optional[int]:
        for i, line in enumerate(lines, start=1):
            low = line.strip()
            if all(tok in low for tok in patterns):
                return i
        return None

    def _find_block_end(start_idx: int) -> Optional[int]:
        # brace counting from start_idx to end of file
        depth = 0
        opened = False
        for i in range(start_idx, len(lines) + 1):
            line = lines[i - 1]
            # Count braces while ignoring those in quotes (very rough)
            # Good-enough heuristic for Java files
            in_str = False
            esc = False
            for ch in line:
                if ch == '"' and not esc:
                    in_str = not in_str
                esc = (ch == '\\' and not esc)
                if in_str:
                    continue
            # Now count braces (naively)
            depth += line.count("{")
            if depth > 0:
                opened = True
            depth -= line.count("}")
            if opened and depth == 0:
                return i
        return None

    # If we have a method name, search for a plausible signature line
    if meth:
        patterns = [meth + "("]
        # Prefer matching class name near-by if we also have it
        if cls_tail:
            # not strict, just increases precision
            pass
        # Include type name tokens to improve hits
        # (but don't require them all to appear on the same line; method name is key)
        start = _find_block_start(patterns)
        if start is not None:
            end = _find_block_end(start) or start
            sig = f"{cls_tail}.{meth}({','.join(short_params)})" if cls_tail else f"{meth}({','.join(short_params)})"
            return start, end, sig
    # Otherwise try the class block
    if cls_tail:
        # look for "class ClassName" or "interface/enum ClassName"
        for kw in ["class", "interface", "enum", "record"]:
            start = _find_block_start([kw, cls_tail])
            if start is not None:
                end = _find_block_end(start) or start
                sig = cls_tail
                return start, end, sig
    return None, None, None

import json as _json
from typing import Optional, List, Literal, Dict, Any

@mcp.tool()
def list_nodes(q: Optional[str] = None,
               node_type: Optional[str] = None,
               community: Optional[object] = None,
               limit: int = 50) -> str:
    """
    Find nodes by optional substring query, type, and community filter.
    """
    df = ENTS
    if node_type:
        df = df[df["type"] == node_type]
    if community is not None and len(COMS):
        row = COMS[COMS["human_readable_id"] == community]
        if len(row) == 0 and isinstance(community, str) and community.isdigit():
            row = COMS[COMS["human_readable_id"] == int(community)]
        if len(row):
            eids = _as_list(row.iloc[0].get("entity_ids"))
            if eids:
                df = df[df["id"].isin(eids)]
    if q:
        ql = q.lower().strip()
        def _contains(s: pd.Series) -> pd.Series:
            return s.fillna("").astype(str).str.lower().str.contains(ql, regex=False)
        df = df[_contains(df["id"]) | _contains(df.get("display_name", pd.Series(dtype=str))) | _contains(df.get("description", pd.Series(dtype=str)))]
    rows = df.head(int(limit))[["id","display_name","type"]].to_dict(orient="records")
    return _json.dumps(rows, indent=2)

@mcp.tool()
def get_node(id: str) -> str:
    """
    Return full node metadata for a given id.
    """
    rows = ENTS[ENTS["id"] == id]
    payload = rows.iloc[0].to_dict() if len(rows) else {}
    # Clean up common artifacts
    desc = payload.get("description")
    if isinstance(desc, str):
        # Drop common NaN artifacts
        desc = re.sub(r"\s*\|\s*attr=nan\b", "", desc, flags=re.I)
        desc = re.sub(r"\battr=nan\b", "", desc, flags=re.I)
        desc = re.sub(r"\breturns=nan\b", "", desc, flags=re.I)
        desc = re.sub(r"\bparams=nan\b", "", desc, flags=re.I)
        payload["description"] = re.sub(r"\s+\|\s+\|\s+", " | ", desc).strip(" |")
    return _json.dumps(_json_sanitize(payload), indent=2)

@mcp.tool()
def neighbors(id: str,
              edge_types: Optional[List[str]] = None,
              direction: Literal["out","in","both"] = "out",
              limit: int = 50) -> str:
    """
    List neighbors/edges of a node.
    """
    if id not in G:
        return _json.dumps([], indent=2)
    items: List[Dict[str, Any]] = []
    types = set(edge_types or [])
    if direction in ("out","both"):
        for _, tgt, ed in G.out_edges(id, data=True):
            if types and ed.get("edge_type") not in types:
                continue
            items.append({"source": id, "target": tgt, "edge_type": ed.get("edge_type"), "description": ed.get("description")})
    if direction in ("in","both"):
        for src, _, ed in G.in_edges(id, data=True):
            if types and ed.get("edge_type") not in types:
                continue
            items.append({"source": src, "target": id, "edge_type": ed.get("edge_type"), "description": ed.get("description")})
    return _json.dumps(items[:int(limit)], indent=2)

@mcp.tool()
def ego(id: str,
        depth: int = 2,
        edge_types: Optional[List[str]] = None,
        max_nodes: int = 200) -> str:
    """
    Return nodes/edges in a bounded ego subgraph via BFS.
    """
    if id not in G:
        return _json.dumps({"nodes":[],"edges":[]}, indent=2)
    if edge_types:
        H = nx.DiGraph((u,v,d) for u,v,d in G.edges(data=True) if d.get("edge_type") in set(edge_types))
    else:
        H = G
    seen = {id}
    frontier = [id]
    for _ in range(int(depth)):
        nxt = []
        for u in frontier:
            for _, v in H.out_edges(u):
                if len(seen) >= int(max_nodes): break
                if v not in seen: seen.add(v); nxt.append(v)
            for v, _ in H.in_edges(u):
                if len(seen) >= int(max_nodes): break
                if v not in seen: seen.add(v); nxt.append(v)
        frontier = nxt
        if not frontier or len(seen) >= int(max_nodes):
            break
    nodes = [{"id": n, "display_name": (G.nodes[n].get("display_name") if n in G.nodes else n), "type": (G.nodes[n].get("type") if n in G.nodes else None)} for n in seen]
    edges = [{"source": u, "target": v, "edge_type": d.get("edge_type"), "description": d.get("description")} for u,v,d in H.subgraph(seen).edges(data=True)]
    return _json.dumps({"nodes":nodes,"edges":edges}, indent=2)

@mcp.tool()
def shortest_path(src: str,
                  dst: str,
                  edge_types: Optional[List[str]] = None,
                  max_len: int = 12) -> str:
    """
    Shortest path src→dst (optionally filtered by edge types).
    """
    if edge_types:
        H = nx.DiGraph((u,v,d) for u,v,d in G.edges(data=True) if d.get("edge_type") in set(edge_types))
    else:
        H = G
    try:
        path = nx.shortest_path(H, src, dst)
    except Exception:
        path = []
    if path and len(path) > int(max_len) + 1:
        return _json.dumps({"error":"path too long","len":len(path)}, indent=2)
    return _json.dumps({"path":path}, indent=2)

@mcp.tool()
def list_communities(level: Optional[int] = None) -> str:
    """
    List communities (optionally by level).
    """
    if len(COMS) == 0:
        return "[]"
    df = COMS
    if level is not None:
        try:
            df = df[df.get("level", 0) == int(level)]
        except Exception:
            pass
    cols = [c for c in ["human_readable_id","community","level","size","title"] if c in df.columns]
    return _json.dumps(df[cols].to_dict(orient="records"), indent=2)

@mcp.tool()
def get_community(id: object) -> str:
    """
    Return entities/relationships for a community (by human_readable_id).
    """
    try:
        cid_int = int(id)
        row = COMS[COMS["human_readable_id"] == cid_int]
    except Exception:
        row = COMS[COMS["human_readable_id"] == id]
    if len(row) == 0:
        return _json.dumps({}, indent=2)
    row = row.iloc[0]
    eids = _as_list(row.get("entity_ids"))
    rids = _as_list(row.get("relationship_ids"))
    # Normalize all to strings for reliable matching
    eids = [str(e) for e in eids if e is not None]
    rids = [str(r) for r in rids if r is not None]
    ents = ENTS[ENTS["id"].astype(str).isin(eids)][["id","display_name","type"]].to_dict(orient="records") if eids else []
    rels = RELS[RELS["id"].astype(str).isin(rids)][["id","source","target","edge_type","description"]].to_dict(orient="records") if rids else []
    return _json.dumps({"entities": ents, "relationships": rels}, indent=2)

@mcp.tool()
def get_community_report(id: object) -> str:
    """
    Return the community report text (advisory; verify before use).
    """
    try:
        cid_int = int(id)
        row = REPS[REPS["human_readable_id"] == cid_int]
    except Exception:
        row = REPS[REPS["human_readable_id"] == id]
    if len(row) == 0:
        return _json.dumps({}, indent=2)
    row = row.iloc[0]
    payload = {"title": row.get("title"), "summary": row.get("summary"), "full_content": row.get("full_content")}
    return _json.dumps(payload, indent=2)

@mcp.tool()
def keyword_search(q: str, limit: int = 50) -> str:
    """
    Keyword search over nodes (id/display_name/description).
    """
    q = (q or "").strip().lower()
    if not q:
        return "[]"
    tokens = [t for t in q.split() if t]
    base = ENTS.copy()
    id_s  = base["id"].fillna("").astype(str).str.lower()
    dn_s  = base.get("display_name", pd.Series(dtype=str)).fillna("").astype(str).str.lower()
    de_s  = base.get("description", pd.Series(dtype=str)).fillna("").astype(str).str.lower()

    mask = pd.Series(True, index=base.index)
    for tok in tokens:
        mask = mask & (id_s.str.contains(tok, regex=False) | dn_s.str.contains(tok, regex=False) | de_s.str.contains(tok, regex=False))

    df = base[mask]
    rows = df.head(int(limit))[["id","display_name","type"]].to_dict(orient="records")
    return _json.dumps(rows, indent=2)

@mcp.tool()
def semantic_search(q: str, top_k: int = 10) -> str:
    """
    Semantic search over text_units (if FAISS index present).
    """
    rows = _semantic_search(q, top_k=int(top_k))
    return _json.dumps(rows, indent=2)


@mcp.tool()
def read_source(id: str, surround: int = 8, full_file: bool = False, max_bytes: int = 500_000) -> str:
    """
    Read source around a node’s location (default), or the entire file.

    Args:
      id: Graph node id to resolve to a source file.
      surround: Number of context lines around the method/class when not `full_file`.
      full_file: When true, return the entire file content (ignores line hints).
      max_bytes: If `full_file` is true and the file exceeds this many bytes, the content
                 is truncated to `max_bytes` and `truncated=true` is returned.
    """
    meta = _resolve_source(id)
    if not meta:
        return _json.dumps({"error":"no source mapping (file not found under allowlist or node has no file/path)"}, indent=2)

    # If caller wants the whole file, read it directly (ignoring line hints)
    if full_file:
        p = pathlib.Path(meta["path"])  # type: ignore[index]
        if p.exists() and p.is_dir():
            return _json.dumps({"error":"path is a directory"}, indent=2)
        if not _in_allowlist(p):
            return _json.dumps({"error":"path not allowlisted"}, indent=2)
        try:
            data = p.read_bytes()
        except Exception as e:
            return _json.dumps({"error":f"read error: {e}"}, indent=2)
        truncated = False
        if len(data) > int(max_bytes):
            data = data[: int(max_bytes)]
            truncated = True
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            # Fallback just in case (shouldn’t happen with errors="replace")
            text = str(data)
        return _json.dumps({
            "path": str(p),
            "signature": meta.get("signature"),
            "truncated": truncated,
            "content": text
        }, indent=2)

    # Otherwise, return a numbered snippet around the known lines; if hints are missing, try to guess
    start_hint = meta.get("start_line")
    end_hint = meta.get("end_line")

    # Treat (1,1) coming from missing metadata as "no hints"; attempt to guess
    def _needs_guess(a, b) -> bool:
        try:
            return (a is None) or (b is None) or (int(a) == 1 and int(b) == 1)
        except Exception:
            return True

    if _needs_guess(start_hint, end_hint):
        gs, ge, gsig = _guess_span(meta["path"], id)
        if gs is not None and ge is not None and ge >= gs:
            start_hint, end_hint = gs, ge
            if gsig:
                meta["signature"] = gsig

    # Normalize to ints (default to file start if still missing)
    try:
        s_hint = int(start_hint) if start_hint is not None else 1
    except Exception:
        s_hint = 1
    try:
        e_hint_candidate = int(end_hint) if end_hint is not None else None
    except Exception:
        e_hint_candidate = None
    e_hint = e_hint_candidate if (e_hint_candidate is not None and e_hint_candidate >= s_hint) else s_hint

    # Expand range by surround on both sides, then slice strictly
    p = pathlib.Path(meta["path"])  # type: ignore[index]
    try:
        total_lines = len(p.read_text(encoding="utf-8", errors="ignore").splitlines())
    except Exception:
        total_lines = 10**9  # fallback high cap

    s_eff = max(1, int(s_hint) - int(surround))
    e_eff = min(total_lines, int(e_hint) + int(surround))

    txt, err = _read_slice(meta["path"], s_eff, e_eff, surround=0)
    if err:
        return _json.dumps({"error": err}, indent=2)

    return _json.dumps({
        "path": meta["path"],
        "signature": _pretty_method_signature(id) or meta.get("signature"),
        "file_length": meta.get("file_length"),
        "start_line": s_hint,
        "end_line": e_hint,
        "effective_start": s_eff,
        "effective_end": e_eff,
        "snippet": txt
    }, indent=2)


# Diagnostic tool: which_source
@mcp.tool()
def which_source(id: str) -> str:
    """
    Diagnostic: show the resolved source path, file length, and any start/end hints.
    """
    meta = _resolve_source(id)
    if not meta:
        return _json.dumps({"error":"no source mapping"}, indent=2)
    return _json.dumps(meta, indent=2)


@mcp.tool()
def read_file_by_node(id: str, max_bytes: int = 500_000) -> str:
    """
    Read the entire source file associated with a graph node id.

    This is a convenience alias for `read_source(full_file=True)`, making it
    obvious to LLMs that the whole file can be retrieved.

    Args:
      id: Graph node id to resolve to a source file.
      max_bytes: If the file exceeds this many bytes, the content is truncated
                 to `max_bytes` and `truncated=true` is returned.
    """
    meta = _resolve_source(id)
    if not meta:
        return _json.dumps({"error":"no source mapping (file not found under allowlist or node has no file/path)"}, indent=2)

    p = pathlib.Path(meta["path"])  # type: ignore[index]
    if p.exists() and p.is_dir():
        return _json.dumps({"error":"path is a directory"}, indent=2)
    if not _in_allowlist(p):
        return _json.dumps({"error":"path not allowlisted"}, indent=2)
    try:
        data = p.read_bytes()
    except Exception as e:
        return _json.dumps({"error":f"read error: {e}"}, indent=2)

    truncated = False
    if len(data) > int(max_bytes):
        data = data[: int(max_bytes)]
        truncated = True
    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        text = str(data)

    return _json.dumps({
        "path": str(p),
        "signature": _pretty_method_signature(id) or meta.get("signature"),
        "truncated": truncated,
        "content": text
    }, indent=2)

@mcp.tool()
def open_file(path: str, start: int = 1, end: Optional[int] = None) -> str:
    """
    Open a file slice (allowlisted).
    """
    if end is None:
        end = start + 200
    txt, err = _read_slice(path, int(start), int(end), surround=0)
    if err:
        return _json.dumps({"error":err}, indent=2)
    return _json.dumps({"path":path,"slice":txt}, indent=2)

# HTTP endpoints - Root endpoint for MCP JSON-RPC protocol
from starlette.responses import StreamingResponse

@app.get("/")
async def mcp_sse_endpoint(request: Request):
    """Handle SSE connection for server notifications"""
    import sys
    print(f"\n[MCP DEBUG] SSE GET request received at /", file=sys.stderr, flush=True)
    print(f"[MCP DEBUG] Headers: {dict(request.headers)}", file=sys.stderr, flush=True)

    async def event_generator():
        """Keep the SSE connection alive and send events when needed"""
        # Send initial connection message
        yield f"event: message\ndata: {json.dumps({'type': 'connection', 'status': 'connected'})}\n\n"

        # Keep connection alive with periodic pings
        while True:
            await asyncio.sleep(30)  # Send keepalive every 30 seconds
            yield f": keepalive\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/")
async def mcp_jsonrpc_root(request: Request):
    """Handle MCP JSON-RPC messages at root"""
    data = await request.json()

    # Debug logging
    import sys
    print(f"\n[MCP DEBUG] Received request:", file=sys.stderr, flush=True)
    print(f"[MCP DEBUG] Headers: {dict(request.headers)}", file=sys.stderr, flush=True)
    print(f"[MCP DEBUG] Body: {json.dumps(data, indent=2)}", file=sys.stderr, flush=True)

    method = data.get("method")
    msg_id = data.get("id")

    if method == "initialize":
        response = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "codegraph-mcp", "version": "1.0.0"}
            }
        }
        print(f"[MCP DEBUG] Response: {json.dumps(response, indent=2)}", file=sys.stderr, flush=True)
        return JSONResponse(content=response)
    elif method == "tools/list":
        tools = await mcp.list_tools()
        tools_list = []
        for tool in tools:
            if hasattr(tool, 'model_dump'):
                tools_list.append(tool.model_dump())
            elif hasattr(tool, 'dict'):
                tools_list.append(tool.dict())
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"tools": tools_list}
        })
    elif method == "tools/call":
        params = data.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        print(f"[MCP DEBUG] Calling tool: {tool_name} with args: {arguments}", file=sys.stderr, flush=True)
        try:
            result = await mcp.call_tool(tool_name, arguments)
            print(f"[MCP DEBUG] Raw tool result type: {type(result)}", file=sys.stderr, flush=True)
            print(f"[MCP DEBUG] Raw tool result: {result}", file=sys.stderr, flush=True)

            if isinstance(result, tuple) and len(result) == 2:
                content_list, result_dict = result
                print(f"[MCP DEBUG] Content list: {content_list}", file=sys.stderr, flush=True)
                print(f"[MCP DEBUG] Result dict: {result_dict}", file=sys.stderr, flush=True)

                # Convert TextContent objects to dictionaries
                serialized_content = []
                for item in content_list:
                    if hasattr(item, 'model_dump'):
                        serialized_content.append(item.model_dump())
                    elif hasattr(item, 'dict'):
                        serialized_content.append(item.dict())
                    else:
                        serialized_content.append(item)

                # MCP protocol expects 'content' to be an array of content items
                response = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": serialized_content
                    }
                }
                print(f"[MCP DEBUG] Sending response: {json.dumps(response, indent=2)}", file=sys.stderr, flush=True)
                return JSONResponse(content=response)
        except Exception as e:
            import traceback
            print(f"[MCP DEBUG] Error calling tool: {e}", file=sys.stderr, flush=True)
            print(f"[MCP DEBUG] Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
            return JSONResponse(content={
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -1, "message": str(e)}
            })

    # Handle notifications (no response needed)
    if method and method.startswith("notifications/"):
        print(f"[MCP DEBUG] Notification received: {method}", file=sys.stderr, flush=True)
        # Notifications don't require a response, just return 200 OK
        return JSONResponse(content={})

    return JSONResponse(content={
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"}
    })

@app.post("/mcp")
async def initialize(request: Request):
    data = await request.json()
    return JSONResponse(content={
        "jsonrpc": "2.0",
        "id": data.get("id"),
        "result": {
            "protocolVersion": "0.1.0",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "codegraph-mcp", "version": "1.0.0"}
        }
    })

@app.post("/mcp/tools/list")
async def list_tools_http():
    # Use FastMCP's built-in list_tools method (async)
    tools = await mcp.list_tools()
    # Convert Tool objects to dictionaries
    tools_list = []
    for tool in tools:
        if hasattr(tool, 'model_dump'):
            # Pydantic v2
            tools_list.append(tool.model_dump())
        elif hasattr(tool, 'dict'):
            # Pydantic v1
            tools_list.append(tool.dict())
        else:
            # Fallback to converting attributes manually
            tools_list.append({
                "name": getattr(tool, 'name', str(tool)),
                "description": getattr(tool, 'description', ''),
                "inputSchema": getattr(tool, 'inputSchema', {})
            })
    return JSONResponse(content={"tools": tools_list})

@app.post("/mcp/tools/call")
async def call_tool_http(request: Request):
    data = await request.json()
    tool_name = data.get("name")
    arguments = data.get("arguments", {})

    try:
        # Use FastMCP's built-in call_tool method (async)
        result = await mcp.call_tool(tool_name, arguments)

        # Extract the actual result from the MCP response
        # FastMCP returns a tuple: (content_list, dict_with_result)
        if isinstance(result, tuple) and len(result) == 2:
            content_list, result_dict = result
            # The second element is the actual result dict
            return JSONResponse(content=result_dict)
        elif hasattr(result, 'model_dump'):
            # Pydantic v2
            result_dict = result.model_dump()
            return JSONResponse(content=result_dict)
        elif hasattr(result, 'dict'):
            # Pydantic v1
            result_dict = result.dict()
            return JSONResponse(content=result_dict)
        else:
            return JSONResponse(content={"result": str(result)})

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


if __name__ == "__main__":
    # Simple runtime diagnostics to stderr
    print(f"[diag] FastMCP starting; py={sys.version}", file=sys.stderr, flush=True)
    # FastMCP v1.13+: run() uses stdio transport by default
    # mcp.run()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )