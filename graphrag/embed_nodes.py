import pickle
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from pyvis.network import Network
import re

def split_identifier(name):
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', name).lower()

# Load graph
with open("domain_graph.gpickle", "rb") as f:
    G = pickle.load(f)

# Load local model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Collect descriptions to embed
text_nodes = []
node_ids = []
for node_id, attrs in G.nodes(data=True):
    raw_desc = attrs.get("description", "").strip()
    if raw_desc:
        desc = raw_desc
    else:
        node_type = attrs.get("type", "UnknownType")
        node_name = attrs.get("name", "Unnamed")
        if node_type == "Method":
            class_name = ".".join(node_id.split(".")[:-1])
            desc = f"method {split_identifier(node_name)} in class {split_identifier(class_name)}"
        elif node_type == "Class":
            desc = f"class {split_identifier(node_name)}"
        else:
            desc = f"{node_type} named {split_identifier(node_name)}"
        attrs["description"] = desc  # Update node with generated description
    if desc:
        node_ids.append(node_id)
        text_nodes.append(desc)
        print(f"Embedding node {node_id} with description: {desc}")
    else:
        print(f"Skipping node {node_id} with no description")

# Embed descriptions
embeddings = model.encode(text_nodes, show_progress_bar=True)

# Attach embeddings
for node_id, embedding in zip(node_ids, embeddings):
    G.nodes[node_id]["embedding"] = embedding.tolist()  # Store as list for JSON/serialization

# Save graph again
with open("domain_graph_embedded.gpickle", "wb") as f:
    pickle.dump(G, f)

print(f"✅ Embedded {len(embeddings)} nodes with descriptions.")

# --- Optional: Visualize with PyVis ---
# Create the PyVis network without notebook mode
net = Network(height="750px", width="100%", directed=True, notebook=False)

# Add nodes
for node_id, attrs in G.nodes(data=True):
    net.add_node(
        node_id,
        label=node_id.split(".")[-1],  # short label
        title=f"Type: {attrs.get('type', '')} Name: {attrs.get('name', '')} Description: {attrs.get('description', '')} File: {attrs.get('file', '')}",
        color="orange" if attrs.get("type") in {"UseCase", "Entity", "Event"} else "lightblue"
    )

# Add edges
for source, target, attrs in G.edges(data=True):
    net.add_edge(source, target, label=attrs.get("type", ""))

# Save and show
net.write_html("graph_embeddded.html")  # <- use write_html instead of show()
print("✅ Graph visualization written to graph.html")