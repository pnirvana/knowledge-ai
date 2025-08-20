import pickle

def load_graph(filepath="domain_graph.gpickle"):
    with open(filepath, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    G = load_graph()
    for node_id, attrs in G.nodes(data=True):
        if node_id.startswith("net.chrisrichardson.ftgo.orderservice.domain.OrderService.updateOrder"):
            print("net.chrisrichardson.ftgo.orderservice.domain.OrderService.updateOrder(long, java.util.function.Function<net.chrisrichardson.ftgo.orderservice.domain.Order, java.util.List<net.chrisrichardson.ftgo.orderservice.api.events.OrderDomainEvent>>)")
            print(f"Node ID: {node_id}, Type: {attrs.get('type', 'unknown')}, Name: {attrs.get('name', 'unknown')}")
    for edge in G.edges(data=True):   
        if edge[1].startswith("net.chrisrichardson.ftgo.orderservice.domain.OrderService.approveOrder"):
            print(f"{edge[0]} → {edge[1]} [{edge[2].get('type', 'unknown')}]")