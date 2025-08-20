import plotly.graph_objs as go
import os
import json
import re
import networkx as nx
import pickle
from pyvis.network import Network
import openai
import random


# --- Config flags (can be overridden via environment variables) ---
INCLUDE_EXTERNAL = os.getenv("INCLUDE_EXTERNAL", "1") not in ("0", "false", "False")
INCLUDE_EDGE_TYPES = set(
    (os.getenv("INCLUDE_EDGE_TYPES") or "calls,creates,publishes_event,has_attribute,defines,reads_attribute,writes_attribute,implemented_by,represented_by")
    .split(",")
)

# Whitelist of external package prefixes to include (others will be filtered out)
EXTERNAL_ALLOWLIST = [s.strip() for s in (os.getenv("EXTERNAL_ALLOWLIST") or 
    "org.springframework.,java.,javax.,com.fasterxml.jackson.,org.apache.kafka.").split(",") if s.strip()]

def external_allowed(node_id: str) -> bool:
    if not is_external_id(node_id):
        return True
    base = node_id[len(EXTERNAL_PREFIX):]
    for pref in EXTERNAL_ALLOWLIST:
        if base.startswith(pref):
            return True
    return False

# Color palettes for nodes and edges
NODE_COLORS = {
    "UseCase": "orange",
    "Entity": "red",
    "Class": "steelblue",
    "Interface": "lightsteelblue",
    "Enum": "lightslategray",
    "Event": "green",
    "Attribute": "lightgreen",
    "Method": "purple",
    "ExternalClass": "#bdbdbd",      # gray
    "ExternalMethod": "#bdbdbd",     # gray
}

EDGE_COLORS = {
    "calls": "#2ca02c",             # green
    "creates": "#ff99c8",           # pink
    "publishes_event": "#d62728",    # red
    "has_attribute": "#ff7f0e",      # orange
    "defines": "#9e9e9e",            # gray
    "reads_attribute": "#1f77b4",    # blue
    "writes_attribute": "#8c564b",   # brown
    "implemented_by": "#9467bd",     # purple
    "represented_by": "#17becf",     # teal
}

EXTERNAL_PREFIX = "ext://"

def is_external_id(node_id: str) -> bool:
    return isinstance(node_id, str) and node_id.startswith(EXTERNAL_PREFIX)

def short_name_from_id(nid: str) -> str:
    # For labels: prefer provided name; if missing, derive from id
    try:
        base = nid
        if "#" in base:
            base = base.split("#", 1)[0]
        if "(" in base:
            base = base[:base.index("(")]
        return base.rsplit(".", 1)[-1]
    except Exception:
        return nid


# --- Load files ---
with open("graph_nodes.json") as f:
    nodes = json.load(f)
    print(f"Loaded {len(nodes)} nodes")

with open("graph_edges.json") as f:
    edges = json.load(f)
    print(f"Loaded {len(edges)} edges")

G = nx.DiGraph()

# --- Step 2: Add static code nodes ---
print("✅ Building static graph")
for node in nodes:
    node_type = node.get("type", "UnknownType") or "UnknownType"
    node_name = node.get("name", "Unnamed") or "Unnamed"
    node_id = node.get("id", "")

    # Skip synthetic nodes like raw "new SomeCommand(...)"
    if node_type == "None" or "new " in node_id or node_name == "Unnamed":
        print(f"⚠️ Skipping synthetic or malformed node: {node_id}")
        continue

    # Check for @Entity annotation
    annotations = node.get("annotations", [])
    if "Entity" in annotations:
        node["type"] = "Entity"
    else:
        node["type"] = node_type

    # Keep attribute names clean; show types only in tooltips/details
    node["name"] = node_name

    G.add_node(node_id, **node)

# --- Step 3: Add static edges ---
for edge in edges:
    edge_type = edge.get("type")
    if edge_type not in INCLUDE_EDGE_TYPES:
        continue

    src = edge.get("from")
    dst = edge.get("to")

    # Filter out edges to disallowed external packages
    if (is_external_id(src) and not external_allowed(src)) or (is_external_id(dst) and not external_allowed(dst)):
        # Skip this edge entirely
        continue

    # Optionally skip external endpoints entirely
    if not INCLUDE_EXTERNAL and (is_external_id(src) or is_external_id(dst)):
        continue

    # Ensure source exists; if external, optionally stub
    if not G.has_node(src):
        if is_external_id(src) and INCLUDE_EXTERNAL and external_allowed(src):
            # Guess external kind by presence of signature
            kind = "ExternalMethod" if "(" in src else "ExternalClass"
            G.add_node(src, id=src, type=kind, name=short_name_from_id(src))
        else:
            print(f"⚠️ Skipping edge (missing source): {src} → {dst}")
            continue

    # Ensure target exists; if external, optionally stub
    if not G.has_node(dst):
        if is_external_id(dst) and INCLUDE_EXTERNAL and external_allowed(dst):
            kind = "ExternalMethod" if "(" in dst else "ExternalClass"
            G.add_node(dst, id=dst, type=kind, name=short_name_from_id(dst))
        else:
            print(f"⚠️ Skipping edge (missing target): {src} → {dst}")
            continue

    G.add_edge(src, dst, type=edge_type)
    if edge_type == "implemented_by":
        print(f"🔧 IMPLEMENTED_BY: {src} → {dst}")

# Prune disallowed or orphaned external nodes
to_remove = []
for nid, data in G.nodes(data=True):
    if is_external_id(nid):
        if not external_allowed(nid):
            to_remove.append(nid)
        else:
            # remove externals with zero degree to keep graph tidy
            if G.degree(nid) == 0:
                to_remove.append(nid)
if to_remove:
    G.remove_nodes_from(to_remove)
    print(f"🧹 Pruned {len(to_remove)} external nodes (disallowed or orphaned)")

# --- Step 4: Add domain nodes ---
print("✅ Adding domain nodes")

print(f"✅ Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

print(f"✅ Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# --- Optional: Enrich graph using LLM ---
print("🤖 Enriching with LLM...")
openai.api_key = os.getenv("OPENAI_API_KEY")

def traverse_for_usecase(G, start_node, max_depth=5):
    visited = set()
    queue = [(start_node, 0)]
    path = []

    while queue:
        nid, depth = queue.pop(0)
        if nid in visited or depth > max_depth:
            continue
        visited.add(nid)
        path.append(nid)

        data = G.nodes[nid]
        if data.get("type") == "Entity":
            break

        for _, neighbor in G.out_edges(nid):
            edge_type = G[nid][neighbor].get("type")
            if edge_type == "publishes_event":
                path.append(neighbor)
                return path  # event published, return early
            if edge_type == "calls":
                queue.append((neighbor, depth + 1))
    return path

known_use_cases = ["CancelOrder", "CreateOrder", "ReviseOrder"]
total_calls = 0

for nid, data in G.nodes(data=True):
    if data.get("type") != "Method":
        continue

    # Only consider adapter layer classes
    fqcn = nid.split("(")[0]  # remove method signature
    if not any(pkg in fqcn for pkg in [".web.", ".grpc.", ".messaging."]):
        continue

    path = traverse_for_usecase(G, nid)
    if len(path) <= 1:
        continue

    context_lines = [f"{G.nodes[n].get('type', '')}: {G.nodes[n].get('name', '')}" for n in path]

    prompt = (
        "The following is a call chain starting from a Java method in the adapter layer. "
        "Based on the names and types, which of the following use cases does it most likely implement:\n"
        f"{', '.join(known_use_cases)}?\n\n"
        "Call Chain:\n" + "\n".join(context_lines) + "\n\n"
        "Answer with only the name of the use case."
    )

    total_calls += 1
    # try:
    #     response = openai.ChatCompletion.create(
    #         model="gpt-4",
    #         messages=[{"role": "user", "content": prompt}],
    #         temperature=0,
    #         max_tokens=20
    #     )
    #     label = response["choices"][0]["message"]["content"].strip()
    #     if label in known_use_cases:
    #         if G.has_node(label):
    #             G.add_edge(nid, label, type="implements")
    #             print(f"✅ {nid} → implements → {label}")
    # except Exception as e:
    #     print(f"⚠️ LLM error for node {nid}: {e}")

print(f"✅ Enriched {total_calls} adapter methods with LLM")
# --- Optional: Save to file ---
with open("domain_graph.gpickle", "wb") as f:
    pickle.dump(G, f)

# --- Optional: Visualize with PyVis ---
# Create the PyVis network without notebook mode
net = Network(height="750px", width="100%", directed=True, notebook=False)

# Slightly stabilize layout for readability
net.barnes_hut()
net.set_options('''{
  "physics": {"stabilization": {"iterations": 200}},
  "edges": {"font": {"size": 10}, "smooth": {"type": "dynamic"}}
}''')

# Add nodes
for node_id, attrs in G.nodes(data=True):
    ntype = attrs.get("type", "")
    node_color = NODE_COLORS.get(ntype, "lightblue")

    # Build rich HTML tooltip
    title_parts = []
    title_parts.append(f"<b>Type</b>: {ntype}")
    title_parts.append(f"<b>Name</b>: {attrs.get('name','')}")
    if attrs.get("description"):
        title_parts.append(f"<b>Description</b>: {attrs.get('description')}")
    if attrs.get("file"):
        s = attrs.get("startLine"); e = attrs.get("endLine")
        span = f" (L{s}-{e})" if s and e else ""
        title_parts.append(f"<b>File</b>: {attrs.get('file')}{span}")
    if ntype == "Attribute":
        title_parts.append(f"<b>AttrType</b>: {attrs.get('attr_type','')}")
    if ntype == "Method":
        params = attrs.get("parameters", [])
        return_type = attrs.get("return_type", "")
        if return_type:
            title_parts.append(f"<b>Returns</b>: {return_type}")
        if params:
            title_parts.append(f"<b>Parameters</b>: {', '.join(params)}")
    if attrs.get("annotations"):
        anns = attrs.get("annotations")
        if isinstance(anns, list):
            title_parts.append(f"<b>Annotations</b>: {', '.join(anns)}")

    title = "<br>".join(title_parts)

    # Clean label: methods show only simple name; others use provided name
    if ntype == "Method":
        method_name = attrs.get("name", "")
        simple_name = method_name.split("(")[0] if "(" in method_name else method_name
        label = simple_name or short_name_from_id(node_id)
    else:
        label = attrs.get("name") or short_name_from_id(node_id)

    net.add_node(node_id, label=label, title=title, color=node_color)

# Add edges
for source, target, attrs in G.edges(data=True):
    etype = attrs.get("type", "")
    if etype not in INCLUDE_EDGE_TYPES:
        continue
    color = EDGE_COLORS.get(etype, "gray")
    # Optional: make external edges dashed
    dashes = is_external_id(source) or is_external_id(target)
    net.add_edge(source, target, label=etype, color=color, physics=True, smooth=True, dashes=dashes)

# Save and show
net.write_html("graph.html")  # <- use write_html instead of show()


print("✅ Graph visualization written to graph.html")

# --- Optional: Generate an interactive Sigma.js explorer with search & filters ---
print("✅ Generating Sigma.js explorer (search + filters)...")

# Compute 2D positions for Sigma (Sigma requires x/y per node)
LAYOUT = os.getenv("SIGMA_LAYOUT", "spring").lower()
try:
    if LAYOUT == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif LAYOUT == "circular":
        pos = nx.circular_layout(G)
    elif LAYOUT == "random":
        pos = {n: (random.random(), random.random()) for n in G.nodes()}
    else:  # default spring layout
        pos = nx.spring_layout(G, seed=42, k=None, iterations=50)
except Exception as e:
    print(f"⚠️ Layout failed ({e}); falling back to random positions")
    pos = {n: (random.random(), random.random()) for n in G.nodes()}

# 1) Project NetworkX graph -> lightweight JSON for the browser
edge_type_set = sorted({data.get("type", "") for _, _, data in G.edges(data=True) if data.get("type")})

def nx_to_sigma_data(G, pos):
    nodes_out = []
    for nid, data in G.nodes(data=True):
        ntype = data.get("type", "")
        nodes_out.append({
            "id": nid,
            "label": ((data.get("name") or short_name_from_id(nid)).split("(")[0] if (data.get("type", "") == "Method") else (data.get("name") or short_name_from_id(nid))),
            "type": ntype,
            "color": NODE_COLORS.get(ntype, "#87CEFA"),
            "x": float(pos.get(nid, (0.0, 0.0))[0]),
            "y": float(pos.get(nid, (0.0, 0.0))[1]),
            "external": bool(is_external_id(nid)),
            "file": data.get("file"),
            "startLine": data.get("startLine"),
            "endLine": data.get("endLine"),
            "attr_type": data.get("attr_type"),
            "return_type": data.get("return_type"),
            "parameters": data.get("parameters", []),
            "annotations": data.get("annotations", []),
            "description": data.get("description", ""),
            # size: degree-based default (tuned in browser)
            "size": max(2, G.degree(nid)),
        })
    edges_out = []
    eid = 0
    for u, v, data in G.edges(data=True):
        et = data.get("type", "")
        edges_out.append({
            "id": f"e{eid}",
            "source": u,
            "target": v,
            "type": et,
            "color": EDGE_COLORS.get(et, "#999"),
            "external": bool(is_external_id(u) or is_external_id(v)),
        })
        eid += 1
    return {"nodes": nodes_out, "edges": edges_out}

sigma_data = nx_to_sigma_data(G, pos)

# 2) Write a single self-contained HTML file (loads data from an embedded <script> tag)
# Precompute embedded JSON to avoid backslash in f-string expression
html_path = "graph_explorer.html"
edge_types_js = json.dumps(edge_type_set)
# Precompute embedded JSON to avoid backslash in f-string expression
embedded_json = json.dumps(sigma_data).replace('</', '<\\/')


# Use a plain template with placeholders to avoid f-string interpolation issues
html_template = """
<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
<title>Graph Explorer</title>
<style>
  html, body { height: 100%; margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
  #app { display: grid; grid-template-columns: 320px 1fr; grid-template-rows: 100%; height: 100%; }
  #sidebar { padding: 12px; border-right: 1px solid #eee; overflow: auto; }
  #sigma-container { height: 100%; }
  .section { margin-bottom: 16px; }
  .badge { display:inline-block; padding:2px 6px; border-radius: 4px; background:#f0f0f0; margin:2px 4px 0 0; font-size: 12px; }
  input[type=search] { width: 100%; padding: 6px 8px; }
  details summary { cursor: pointer; }
  .muted { color: #666; font-size: 12px; }
</style>
<!-- Graphology + Sigma.js -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/graphology/0.25.4/graphology.umd.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/sigma.js/2.4.0/sigma.min.js"></script>
</head>
<body>
<div id=\"app\">
  <div id=\"sidebar\">
    <div class=\"section\">
      <h3>Search</h3>
      <input id=\"q\" type=\"search\" placeholder=\"Search nodes by label, type, or annotation...\" />
      <div class=\"muted\">Press Enter to focus; Esc to clear.</div>
    </div>
    <div class=\"section\">
      <h3>Filters</h3>
      <label><input type=\"checkbox\" id=\"toggle-external\" checked /> Show external</label>
      <details open>
        <summary>Edge types</summary>
        <div id=\"edge-types\"></div>
      </details>
      <details>
        <summary>Node types</summary>
        <div id=\"node-types\"></div>
      </details>
    </div>
    <div class=\"section\">
      <h3>Selection</h3>
      <div id=\"selection\" class=\"muted\">Click a node to see details</div>
    </div>
  </div>
  <div id=\"sigma-container\"></div>
</div>

<!-- Embedded data -->
<script id=\"graph-data\" type=\"application/json\">__EMBEDDED_JSON__</script>
<script>
(function() {
  const RAW = JSON.parse(document.getElementById('graph-data').textContent);
  const ALLOWED_EDGE_TYPES = new Set(__EDGE_TYPES__);

  // Build graphology graph
  const Graph = window.graphology;
  const graph = new Graph.MultiDirectedGraph();

  RAW.nodes.forEach(n => graph.addNode(n.id, n));
  RAW.edges.forEach(e => {
    if (!ALLOWED_EDGE_TYPES.has(e.type)) return;
    graph.addEdge(e.source, e.target, e);
  });

  // Initialize hidden flags
  graph.forEachNode((id) => graph.setNodeAttribute(id, 'hidden', false));
  graph.forEachEdge((eid) => graph.setEdgeAttribute(eid, 'hidden', false));

  // Renderer
  const container = document.getElementById('sigma-container');
  const renderer = new Sigma(graph, container, {
    renderEdgeLabels: false,
    minEdgeSize: 0.5,
    maxEdgeSize: 2.5,
    minNodeSize: 2,
    maxNodeSize: 12,
  });

  // Reducer state
  let highlightedNode = null;

  // Node & edge reducers honor hidden flags and highlight selection neighborhood
  renderer.setSetting('nodeReducer', (node, data) => {
    const res = { ...data };
    if (graph.getNodeAttribute(node, 'hidden')) res.hidden = true;
    if (highlightedNode) {
      if (node !== highlightedNode && !graph.areNeighbors(node, highlightedNode)) {
        res.color = '#ddd';
      }
    }
    return res;
  });
  renderer.setSetting('edgeReducer', (edge, data) => {
    const res = { ...data };
    if (graph.getEdgeAttribute(edge, 'hidden')) res.hidden = true;
    if (highlightedNode) {
      const [s, t] = graph.extremities(edge);
      if (s !== highlightedNode && t !== highlightedNode) {
        res.color = '#ddd';
      }
    }
    return res;
  });

  // Optional: if ForceAtlas2 is available on the page, run a few iterations
  if (window.fa2 && typeof window.fa2.assign === 'function') {
    try { window.fa2.assign(graph, { iterations: 200, settings: { gravity: 0.5, scalingRatio: 10 } }); } catch (e) {}
  }

  // Build filter controls
  const edgeTypesDiv = document.getElementById('edge-types');
  ALLOWED_EDGE_TYPES.forEach(t => {
    const id = 'edge-' + t;
    const row = document.createElement('label');
    row.innerHTML = `<input type="checkbox" id="${id}" checked /> ${t}`;
    edgeTypesDiv.appendChild(row);
    edgeTypesDiv.appendChild(document.createElement('br'));
  });

  const nodeTypesDiv = document.getElementById('node-types');
  const NODE_TYPES = new Set();
  graph.forEachNode((id, a) => NODE_TYPES.add(a.type||''));
  NODE_TYPES.forEach(t => {
    const id = 'node-' + t;
    const row = document.createElement('label');
    row.innerHTML = `<input type="checkbox" id="${id}" checked /> ${t||'(none)'} `;
    nodeTypesDiv.appendChild(row);
    nodeTypesDiv.appendChild(document.createElement('br'));
  });

  // Filtering state
  const state = {
    showExternal: true,
    edgeTypes: new Set(ALLOWED_EDGE_TYPES),
    nodeTypes: new Set(Array.from(NODE_TYPES)),
    q: ''
  };

  function applyFilters() {
    const q = state.q.toLowerCase().trim();
    const showExternal = state.showExternal;

    // Edge visibility via hidden attr
    graph.forEachEdge((eid, attr, s, t, u) => {
      const typeOk = state.edgeTypes.has(attr.type);
      const externalOk = showExternal || !attr.external;
      const visible = typeOk && externalOk;
      graph.setEdgeAttribute(eid, 'hidden', !visible);
    });

    // Node visibility via hidden attr
    graph.forEachNode((id, a) => {
      const typeOk = state.nodeTypes.has(a.type||'');
      const externalOk = showExternal || !a.external;
      const text = (a.label||'').toLowerCase() + ' ' + (a.type||'').toLowerCase() + ' ' + (a.annotations||[]).join(' ').toLowerCase();
      const searchOk = !q || text.includes(q);
      const visible = typeOk && externalOk && searchOk;
      graph.setNodeAttribute(id, 'hidden', !visible);
    });

    renderer.refresh();
  }

  // Wire inputs
  document.getElementById('toggle-external').addEventListener('change', (e) => {
    state.showExternal = e.target.checked; applyFilters();
  });

  ALLOWED_EDGE_TYPES.forEach(t => {
    document.getElementById('edge-' + t).addEventListener('change', (e) => {
      if (e.target.checked) state.edgeTypes.add(t); else state.edgeTypes.delete(t);
      applyFilters();
    });
  });

  NODE_TYPES.forEach(t => {
    document.getElementById('node-' + t).addEventListener('change', (e) => {
      if (e.target.checked) state.nodeTypes.add(t); else state.nodeTypes.delete(t);
      applyFilters();
    });
  });

  const q = document.getElementById('q');
  q.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { q.value = ''; state.q=''; applyFilters(); }
    if (e.key === 'Enter') { state.q = q.value; applyFilters(); }
  });

  // Selection panel
  const sel = document.getElementById('selection');
  renderer.on('clickNode', ({node}) => {
    const a = graph.getNodeAttributes(node);
    const meta = [];
    meta.push(`<div><b>${a.label}</b></div>`);
    meta.push(`<div class="badge">type: ${a.type||''}</div>`);
    if (a.external) meta.push(`<div class="badge">external</div>`);
    if (a.file) meta.push(`<div>file: ${a.file} ${(a.startLine?`(L${a.startLine}-${a.endLine||''})`:'' )}</div>`);
    if (a.return_type) meta.push(`<div>returns: ${a.return_type}</div>`);
    if (a.parameters && a.parameters.length) meta.push(`<div>params: ${a.parameters.join(', ')}</div>`);
    if (a.attr_type) meta.push(`<div>attr type: ${a.attr_type}</div>`);
    if (a.annotations && a.annotations.length) meta.push(`<div>annotations: ${a.annotations.join(', ')}</div>`);
    if (a.description) meta.push(`<div style="margin-top:6px;">${a.description}</div>`);
    sel.innerHTML = meta.join('');

    highlightedNode = node;
    renderer.refresh();
  });

  renderer.on('clickStage', () => {
    highlightedNode = null;
    sel.innerHTML = '<span class="muted">Click a node to see details</span>';
    renderer.refresh();
  });

  applyFilters();
})();
<\/script>
<!-- Note: browsers may try to request /favicon.ico or .js.map files on file://; it's safe to ignore 404s and source map warnings in the console. -->
</body>
</html>
"""

# Fill placeholders
html = (html_template
        .replace("__EMBEDDED_JSON__", embedded_json)
        .replace("__EDGE_TYPES__", edge_types_js))

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)


print(f"✅ Sigma.js explorer written to {html_path}")

# --- Optional: Generate a Cytoscape.js explorer (robust offline) ---
print("✅ Generating Cytoscape.js explorer (search + filters)...")

cy_nodes = []
for nid, data in G.nodes(data=True):
    ntype = data.get("type", "")
    # Compute label and signature for methods
    raw_label = (data.get("name") or short_name_from_id(nid))
    label = raw_label.split("(")[0] if ntype == "Method" else raw_label
    params_list = data.get("parameters", [])
    # Ensure params as strings
    params_list = [str(p) for p in params_list] if isinstance(params_list, list) else []
    return_type = data.get("return_type") or ""
    signature = ""
    if ntype == "Method":
        signature = f"{label}({', '.join(params_list)})"
        if return_type:
            signature += f" : {return_type}"

    cy_nodes.append({
        "data": {
            "id": nid,
            "label": label,
            "type": ntype,
            "external": bool(is_external_id(nid)),
            "color": NODE_COLORS.get(ntype, "#87CEFA"),
            "file": data.get("file"),
            "startLine": data.get("startLine"),
            "endLine": data.get("endLine"),
            "attr_type": data.get("attr_type"),
            "return_type": return_type,
            "parameters": ", ".join(params_list),
            "signature": signature,
            "annotations": ", ".join(data.get("annotations", [])) if isinstance(data.get("annotations"), list) else (data.get("annotations") or ""),
            "description": data.get("description", ""),
        }
    })

cy_edges = []
_eid = 0
for u, v, data in G.edges(data=True):
    et = data.get("type", "")
    if et not in INCLUDE_EDGE_TYPES:
        continue
    # Skip disallowed externals per allowlist
    if (is_external_id(u) and not external_allowed(u)) or (is_external_id(v) and not external_allowed(v)):
        continue
    cy_edges.append({
        "data": {
            "id": f"e{_eid}",
            "source": u,
            "target": v,
            "type": et,
            "external": bool(is_external_id(u) or is_external_id(v)),
            "color": EDGE_COLORS.get(et, "#999"),
        }
    })
    _eid += 1

cy_data = {"nodes": cy_nodes, "edges": cy_edges}
cy_embedded = json.dumps(cy_data).replace('</', '<\/')
cy_edge_types_js = json.dumps(sorted(list(INCLUDE_EDGE_TYPES)))

cy_path = "graph_cyto.html"
cy_html = """
<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
<title>Graph Explorer (Cytoscape)</title>
<style>
  html, body { height: 100%; margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
  #app { display: grid; grid-template-columns: 320px 1fr; grid-template-rows: 100%; height: 100%; }
  #sidebar { padding: 12px; border-right: 1px solid #eee; overflow: auto; }
  #cy { height: 100%; width: 100%; }
  .section { margin-bottom: 16px; }
  .badge { display:inline-block; padding:2px 6px; border-radius: 4px; background:#f0f0f0; margin:2px 4px 0 0; font-size: 12px; }
  input[type=search] { width: 100%; padding: 6px 8px; }
  details summary { cursor: pointer; }
  .muted { color: #666; font-size: 12px; }
</style>
<!-- Cytoscape.js UMD -->
<script src=\"https://cdn.jsdelivr.net/npm/cytoscape@3.28.1/dist/cytoscape.min.js\"></script>
</head>
<body>
<div id=\"app\">
  <div id=\"sidebar\">
    <div class=\"section\">
      <h3>Search</h3>
      <input id=\"q\" type=\"search\" placeholder=\"Search nodes by label, type, or annotation...\" />
      <div class=\"muted\">Press Enter to apply; Esc to clear.</div>
    </div>
    <div class=\"section\">
      <h3>Filters</h3>
      <label><input type=\"checkbox\" id=\"toggle-external\" checked /> Show external</label>
      <details open>
        <summary>Edge types</summary>
        <div id=\"edge-types\"></div>
      </details>
      <details>
        <summary>Node types</summary>
        <div id=\"node-types\"></div>
      </details>
    </div>

    <div class=\"section\">
      <h3>Focus</h3>
      <div style=\"margin-bottom:6px;\">Depth: <input id=\"focus-depth\" type=\"range\" min=\"1\" max=\"5\" value=\"2\" /> <span id=\"depth-val\">2</span></div>
      <div style=\"margin-bottom:6px;\">Direction:
        <label><input type=\"radio\" name=\"dir\" value=\"both\" checked /> both</label>
        <label><input type=\"radio\" name=\"dir\" value=\"out\" /> out</label>
        <label><input type=\"radio\" name=\"dir\" value=\"in\" /> in</label>
      </div>
      <button id=\"btn-focus\" disabled>Focus on selection</button>
      <button id=\"btn-clear-focus\" disabled>Clear focus</button>
      <div class=\"muted\">Select a node, then click Focus.</div>
    </div>

    <div class=\"section\">
      <h3>Selection</h3>
      <div id=\"selection\" class=\"muted\">Click a node to see details</div>
    </div>
  </div>
  <div id=\"cy\"></div>
</div>

<script id=\"graph-data\" type=\"application/json\">__DATA__</script>
<script>
(function(){
  const RAW = JSON.parse(document.getElementById('graph-data').textContent);
  const ALLOWED_EDGE_TYPES = new Set(__EDGE_TYPES__);

  // Cytoscape init
  const cy = cytoscape({
    container: document.getElementById('cy'),
    elements: RAW,
    style: [
      { selector: 'node', style: { 'label': 'data(label)', 'background-color': 'data(color)', 'font-size': 8 } },
      { selector: 'edge', style: { 'curve-style': 'bezier', 'target-arrow-shape': 'triangle', 'width': 1, 'line-color': 'data(color)', 'target-arrow-color': 'data(color)', 'font-size': 6, 'label': 'data(type)' } },
      { selector: 'node[external = \"true\"]', style: { 'opacity': 0.7, 'shape': 'round-rectangle' } },
      { selector: 'edge[external = \"true\"]', style: { 'line-style': 'dashed' } }
    ],
    layout: { name: 'cose', animate: false }
  });

  // Build filter controls
  const nodeTypes = new Set();
  cy.nodes().forEach(n => nodeTypes.add(n.data('type')||''));

  const edgeTypesDiv = document.getElementById('edge-types');
  ALLOWED_EDGE_TYPES.forEach(t => {
    const id = 'edge-' + t;
    const row = document.createElement('label');
    row.innerHTML = `<input type="checkbox" id="${id}" checked /> ${t}`;
    edgeTypesDiv.appendChild(row);
    edgeTypesDiv.appendChild(document.createElement('br'));
  });

  const nodeTypesDiv = document.getElementById('node-types');
  nodeTypes.forEach(t => {
    const id = 'node-' + t;
    const row = document.createElement('label');
    row.innerHTML = `<input type="checkbox" id="${id}" checked /> ${t||'(none)'} `;
    nodeTypesDiv.appendChild(row);
    nodeTypesDiv.appendChild(document.createElement('br'));
  });

  let currentSelectedId = null;
  let focusNodes = new Set();
  let focusEdges = new Set();

  const state = {
    showExternal: true,
    edgeTypes: new Set(ALLOWED_EDGE_TYPES),
    nodeTypes: new Set(Array.from(nodeTypes)),
    q: '',
    focusDir: 'both',
    focusDepth: 2
  };

  function applyFilters(){
    const q = state.q.toLowerCase().trim();
    const showExternal = state.showExternal;
    const hasFocus = focusNodes.size > 0;

    // Edges
    cy.edges().forEach(e => {
      const typeOk = state.edgeTypes.has(e.data('type'));
      const externalOk = showExternal || !e.data('external');
      const focusOk = !hasFocus || focusEdges.has(e.id());
      e.style('display', (typeOk && externalOk && focusOk) ? 'element' : 'none');
    });

    // Nodes
    cy.nodes().forEach(n => {
      const typeOk = state.nodeTypes.has(n.data('type')||'');
      const externalOk = showExternal || !n.data('external');
      const text = ((n.data('label')||'') + ' ' + (n.data('type')||'') + ' ' + (n.data('annotations')||'')).toLowerCase();
      const searchOk = !q || text.includes(q);
      const focusOk = !hasFocus || focusNodes.has(n.id());
      n.style('display', (typeOk && externalOk && searchOk && focusOk) ? 'element' : 'none');
    });
  }

  // Focus helpers
  function computeFocus(nodeId, depth, dir){
    focusNodes = new Set();
    focusEdges = new Set();
    const start = cy.getElementById(nodeId);
    if (!start || start.empty()) return;

    let frontier = start;
    let visited = start;
    focusNodes.add(start.id());

    const expandOut = (dir === 'out' || dir === 'both');
    const expandIn  = (dir === 'in'  || dir === 'both');

    for (let i = 0; i < depth; i++) {
      let nextNodes = cy.collection();
      if (expandOut) {
        nextNodes = nextNodes.union(frontier.outgoers('node'));
        frontier.outgoers('edge').forEach(e => focusEdges.add(e.id()));
      }
      if (expandIn) {
        nextNodes = nextNodes.union(frontier.incomers('node'));
        frontier.incomers('edge').forEach(e => focusEdges.add(e.id()));
      }
      nextNodes.forEach(n => focusNodes.add(n.id()));
      frontier = nextNodes.difference(visited);
      visited = visited.union(nextNodes);
      if (frontier.empty()) break;
    }

    // Include edges between focused nodes
    const nodesColl = cy.collection(Array.from(focusNodes).map(id => cy.getElementById(id)));
    nodesColl.edgesWith(nodesColl).forEach(e => focusEdges.add(e.id()));
  }

  // Wire filter inputs
  document.getElementById('toggle-external').addEventListener('change', (e) => { state.showExternal = e.target.checked; applyFilters(); });
  ALLOWED_EDGE_TYPES.forEach(t => {
    document.getElementById('edge-' + t).addEventListener('change', (e) => { if (e.target.checked) state.edgeTypes.add(t); else state.edgeTypes.delete(t); applyFilters(); });
  });
  nodeTypes.forEach(t => {
    document.getElementById('node-' + t).addEventListener('change', (e) => { if (e.target.checked) state.nodeTypes.add(t); else state.nodeTypes.delete(t); applyFilters(); });
  });
  const q = document.getElementById('q');
  q.addEventListener('keydown', (e) => { if (e.key === 'Escape') { q.value=''; state.q=''; applyFilters(); } if (e.key === 'Enter') { state.q=q.value; applyFilters(); } });

  // Focus controls
  const depthInput = document.getElementById('focus-depth');
  const depthVal = document.getElementById('depth-val');
  depthInput.addEventListener('input', () => { state.focusDepth = parseInt(depthInput.value, 10) || 1; depthVal.textContent = String(state.focusDepth); });
  Array.from(document.querySelectorAll('input[name="dir"]')).forEach(r => r.addEventListener('change', (e) => { state.focusDir = e.target.value; }));
  const btnFocus = document.getElementById('btn-focus');
  const btnClear = document.getElementById('btn-clear-focus');

  btnFocus.addEventListener('click', () => {
    if (!currentSelectedId) return;
    computeFocus(currentSelectedId, state.focusDepth, state.focusDir);
    btnClear.disabled = false;
    applyFilters();
  });

  btnClear.addEventListener('click', () => {
    focusNodes.clear();
    focusEdges.clear();
    btnClear.disabled = true;
    applyFilters();
  });

  // Selection panel
  const sel = document.getElementById('selection');
    cy.on('tap', 'node', (evt) => {
      const a = evt.target.data();
      currentSelectedId = a.id;
      btnFocus.disabled = false;

      const meta = [];
      meta.push(`<div><b>${a.label}</b></div>`);
      meta.push(`<div class=\"badge\">type: ${a.type||''}</div>`);
      if (a.external) meta.push(`<div class=\"badge\">external</div>`);
      if (a.file) meta.push(`<div>file: ${a.file} ${(a.startLine?`(L${a.startLine}-${a.endLine||''})`:'' )}</div>`);
      if (a.signature) meta.push(`<div>signature: <code>${a.signature}</code></div>`);
      else {
        if (a.parameters) meta.push(`<div>params: ${a.parameters}</div>`);
        if (a.return_type) meta.push(`<div>returns: ${a.return_type}</div>`);
      }
      if (a.attr_type) meta.push(`<div>attr type: ${a.attr_type}</div>`);
      if (a.annotations) meta.push(`<div>annotations: ${a.annotations}</div>`);
      if (a.description) meta.push(`<div style=\"margin-top:6px;\">${a.description}</div>`);
      sel.innerHTML = meta.join('');
    });

  // Initial apply
  applyFilters();
})();
</script>
</body>
</html>
"""

with open(cy_path, "w", encoding="utf-8") as f:
    f.write(cy_html.replace("__DATA__", cy_embedded).replace("__EDGE_TYPES__", cy_edge_types_js))

print(f"✅ Cytoscape.js explorer written to {cy_path}")
