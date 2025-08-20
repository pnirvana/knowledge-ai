import os
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load graph with embeddings
with open("domain_graph_embedded.gpickle", "rb") as f:
    G = pickle.load(f)

# Load same model used for embedding
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Semantic Query ---
query = "cancel an order and notify other services"
query_vec = model.encode([query])

# --- Similarity Search ---
scores = []
for node_id, attrs in G.nodes(data=True):
    emb = attrs.get("embedding")
    if emb:
        sim = cosine_similarity([query_vec[0]], [emb])[0][0]
        scores.append((sim, node_id))

# --- Show Top Matches ---
print(f"\n🔍 Query: \"{query}\"\n")
print("Top 5 matching nodes:\n")
for score, node_id in sorted(scores, reverse=True)[:5]:
    node = G.nodes[node_id]
    print(f"{score:.2f} → {node_id} [{node.get('type')}]")
    print(f"    → {node.get('description')}\n")
def visualize_subgraph(subgraph, filename="subgraph.html"):
    from pyvis.network import Network
    net = Network(height="750px", width="100%", notebook=False, directed=True)
    for node_id, attrs in subgraph.nodes(data=True):
        net.add_node(
            node_id,
            label=attrs.get("label", str(node_id)),
            title=attrs.get("description", ""),
            type=attrs.get("type", ""),
        )
    for source, target, attrs in subgraph.edges(data=True):
        net.add_edge(source, target, **attrs)
    net.show_buttons(filter_=['physics'])
    net.write_html(filename)

    # Inject metadata panel into HTML
    with open(filename, "r", encoding="utf-8") as f:
        html = f.read()

    panel_html = """
<div id="info-panel" style="position:fixed; top:10px; right:10px; width:300px; height:90vh; overflow:auto;
background:#f9f9f9; border:1px solid #ccc; padding:10px; font-family:sans-serif; font-size:12px; z-index:9999;">
<h3>Node Details</h3>
<p>Click a node to see its metadata here.</p>
</div>
<script>
  network.on("click", function (params) {
    if (params.nodes.length > 0) {
      var nodeId = params.nodes[0];
      var nodeData = nodes.get(nodeId);
      document.getElementById("info-panel").innerHTML = `
        <h3>${nodeData.label}</h3>
        <p><strong>ID:</strong> ${nodeId}</p>
        <p><strong>Type:</strong> ${nodeData.type || "N/A"}</p>
        <p><strong>Description:</strong><br>${nodeData.title || "No description"}</p>
      `;
    }
  });
</script>
"""

    html = html.replace("</body>", panel_html + "\n</body>")

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)