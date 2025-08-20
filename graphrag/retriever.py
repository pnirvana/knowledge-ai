import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from pyvis.network import Network

def load_graph(filepath="domain_graph_embedded.gpickle"):
    with open(filepath, "rb") as f:
        return pickle.load(f)

def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def find_top_nodes(G, model, query, top_k=3):
    query_vec = model.encode(query)
    scores = []

    for node_id, attrs in G.nodes(data=True):
        emb = attrs.get("embedding")
        if emb:
            sim = cosine_similarity([query_vec], [emb])[0][0]
            scores.append((sim, node_id))

    return sorted(scores, reverse=True)[:top_k]

def collect_context(G, node_id, depth=1):
    context_nodes = set([node_id])
    frontier = set([node_id])

    for _ in range(depth):
        new_frontier = set()
        for nid in frontier:
            new_frontier.update(G.successors(nid))
            new_frontier.update(G.predecessors(nid))
        context_nodes.update(new_frontier)
        frontier = new_frontier

    return list(context_nodes)

def format_nodes(G, node_ids):
    formatted = []
    for nid in node_ids:
        node = G.nodes[nid]
        desc = node.get("description") or node.get("name", "")
        formatted.append(f"{nid} ({node.get('type')}): {desc}")
    return formatted

if __name__ == "__main__":
    G = load_graph()
    model = load_model()

    query = input("🔍 Enter your semantic query:\n> ")
    top_matches = find_top_nodes(G, model, query, top_k=2)

    print("\n🎯 Top Matches:")
    for sim, nid in top_matches:
        print(f"{sim:.2f} → {nid} ({G.nodes[nid].get('type')})")

    print("\n🧠 Context Nodes:\n")
    for _, nid in top_matches:
        context = collect_context(G, nid, depth=1)
        formatted = format_nodes(G, context)
        for line in formatted:
            print("•", line)
        print("-" * 60)

    def visualize_subgraph(G, node_ids, filename="retrieved_context.html"):
        subgraph = G.subgraph(node_ids)
        net = Network(height="750px", width="100%", directed=True)

        for node_id, data in subgraph.nodes(data=True):
            net.add_node(
                node_id,
                label=node_id.split(".")[-1],
                title=data.get("description") or data.get("name", ""),
                color="orange" if data.get("type") in {"UseCase", "Entity", "Event"} else "lightblue"
            )

        for source, target, data in subgraph.edges(data=True):
            net.add_edge(source, target, label=data.get("type", ""))

        net.write_html(filename)
        print(f"\n🖼️  Subgraph visualization written to: {filename}")

    # Call visualization
    all_context_nodes = set()
    for _, nid in top_matches:
        context = collect_context(G, nid, depth=1)
        all_context_nodes.update(context)

    visualize_subgraph(G, list(all_context_nodes), filename="context_all_matches.html")