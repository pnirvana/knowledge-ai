#!/usr/bin/env python3
"""
Robust Graph Visualization Tool for Large Graphs

This tool provides a Python-based interactive graph viewer that can handle
large graphs without browser performance issues. It uses Dash + Plotly for
server-side rendering and progressive loading.

Features:
- Server-side filtering and layout computation
- Progressive/lazy loading of graph neighborhoods
- Multiple layout algorithms (spring, hierarchical, circular)
- Search and filter capabilities
- Export subgraphs
- Better performance for large graphs (1000+ nodes)

Usage:
    python graph_viewer.py [--graph domain_graph.gpickle] [--port 8050]
"""

import argparse
import pickle
import networkx as nx
import plotly.graph_objs as go
from dash import Dash, html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from typing import Set, Dict, Any, List, Tuple
import json

# Color palettes (matching build_graph.py)
NODE_COLORS = {
    "UseCase": "orange",
    "Entity": "red",
    "Class": "steelblue",
    "Interface": "lightsteelblue",
    "Enum": "lightslategray",
    "Event": "green",
    "Attribute": "lightgreen",
    "Method": "purple",
    "ExternalClass": "#bdbdbd",
    "ExternalMethod": "#bdbdbd",
}

EDGE_COLORS = {
    "calls": "#2ca02c",
    "creates": "#ff99c8",
    "publishes_event": "#d62728",
    "has_attribute": "#ff7f0e",
    "defines": "#9e9e9e",
    "reads_attribute": "#1f77b4",
    "writes_attribute": "#8c564b",
    "implemented_by": "#9467bd",
    "represented_by": "#17becf",
}

EXTERNAL_PREFIX = "ext://"


class GraphViewer:
    """Interactive graph viewer with server-side filtering"""

    def __init__(self, graph_path: str):
        print(f"Loading graph from {graph_path}...")
        with open(graph_path, 'rb') as f:
            self.G = pickle.load(f)

        print(f"Loaded graph with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")

        # Precompute node and edge types for filters
        self.node_types = set()
        self.edge_types = set()

        for _, data in self.G.nodes(data=True):
            if 'type' in data:
                self.node_types.add(data['type'])

        for _, _, data in self.G.edges(data=True):
            if 'type' in data:
                self.edge_types.add(data['type'])

        self.node_types = sorted(self.node_types)
        self.edge_types = sorted(self.edge_types)

        # Cache for layouts
        self.layout_cache = {}

        print(f"Node types: {', '.join(self.node_types)}")
        print(f"Edge types: {', '.join(self.edge_types)}")

    def is_external_id(self, node_id: str) -> bool:
        """Check if node is external"""
        return isinstance(node_id, str) and node_id.startswith(EXTERNAL_PREFIX)

    def short_name(self, node_id: str) -> str:
        """Get short name from node ID"""
        try:
            base = node_id
            if "#" in base:
                base = base.split("#", 1)[0]
            if "(" in base:
                base = base[:base.index("(")]
            return base.rsplit(".", 1)[-1]
        except Exception:
            return node_id

    def filter_graph(
        self,
        node_types: Set[str] = None,
        edge_types: Set[str] = None,
        include_external: bool = True,
        search_query: str = "",
        focus_node: str = None,
        focus_depth: int = 2,
        max_nodes: int = 500
    ) -> nx.DiGraph:
        """
        Filter graph based on criteria and return subgraph

        Args:
            node_types: Set of node types to include
            edge_types: Set of edge types to include
            include_external: Whether to include external nodes
            search_query: Search string for node labels
            focus_node: Node ID to focus on (neighborhood view)
            focus_depth: Depth for neighborhood extraction
            max_nodes: Maximum nodes to return (for performance)
        """
        if node_types is None:
            node_types = set(self.node_types)
        if edge_types is None:
            edge_types = set(self.edge_types)

        # Start with focus neighborhood if specified
        if focus_node and focus_node in self.G:
            # Get ego graph (neighborhood) around focus node
            subgraph = nx.ego_graph(self.G, focus_node, radius=focus_depth, undirected=False)
        else:
            subgraph = self.G.copy()

        # Filter nodes
        nodes_to_remove = []
        for node_id, data in subgraph.nodes(data=True):
            # Type filter
            if data.get('type', '') not in node_types:
                nodes_to_remove.append(node_id)
                continue

            # External filter
            if not include_external and self.is_external_id(node_id):
                nodes_to_remove.append(node_id)
                continue

            # Search filter
            if search_query:
                name = data.get('name', '')
                label = self.short_name(node_id)
                if search_query.lower() not in name.lower() and search_query.lower() not in label.lower():
                    nodes_to_remove.append(node_id)
                    continue

        subgraph.remove_nodes_from(nodes_to_remove)

        # Filter edges
        edges_to_remove = []
        for u, v, data in subgraph.edges(data=True):
            if data.get('type', '') not in edge_types:
                edges_to_remove.append((u, v))

        subgraph.remove_edges_from(edges_to_remove)

        # Limit size for performance
        if subgraph.number_of_nodes() > max_nodes:
            # Keep highest degree nodes
            nodes_by_degree = sorted(
                subgraph.nodes(),
                key=lambda n: subgraph.degree(n),
                reverse=True
            )
            keep_nodes = set(nodes_by_degree[:max_nodes])
            if focus_node:
                keep_nodes.add(focus_node)

            remove_nodes = set(subgraph.nodes()) - keep_nodes
            subgraph.remove_nodes_from(remove_nodes)

            print(f"Limited graph to {max_nodes} highest-degree nodes")

        return subgraph

    def compute_layout(self, subgraph: nx.DiGraph, layout_type: str = "spring") -> Dict[str, Tuple[float, float]]:
        """Compute node positions using specified layout algorithm"""

        cache_key = (tuple(sorted(subgraph.nodes())), layout_type)
        if cache_key in self.layout_cache:
            return self.layout_cache[cache_key]

        if subgraph.number_of_nodes() == 0:
            return {}

        print(f"Computing {layout_type} layout for {subgraph.number_of_nodes()} nodes...")

        try:
            if layout_type == "spring":
                pos = nx.spring_layout(subgraph, seed=42, k=1.5, iterations=50)
            elif layout_type == "kamada_kawai":
                pos = nx.kamada_kawai_layout(subgraph)
            elif layout_type == "circular":
                pos = nx.circular_layout(subgraph)
            elif layout_type == "hierarchical":
                # Try to use graphviz for hierarchical layout
                try:
                    pos = nx.nx_agraph.graphviz_layout(subgraph, prog='dot')
                except:
                    # Fallback to shell layout if graphviz not available
                    pos = nx.shell_layout(subgraph)
            else:
                pos = nx.spring_layout(subgraph, seed=42)
        except Exception as e:
            print(f"Layout failed: {e}, using random layout")
            import random
            pos = {n: (random.random(), random.random()) for n in subgraph.nodes()}

        self.layout_cache[cache_key] = pos
        return pos

    def create_plotly_figure(
        self,
        subgraph: nx.DiGraph,
        layout_type: str = "spring",
        show_labels: bool = True,
        show_edge_labels: bool = False
    ) -> go.Figure:
        """Create Plotly figure from subgraph"""

        pos = self.compute_layout(subgraph, layout_type)

        # Create edge traces
        edge_traces = []
        for u, v, data in subgraph.edges(data=True):
            if u not in pos or v not in pos:
                continue

            x0, y0 = pos[u]
            x1, y1 = pos[v]

            edge_type = data.get('type', '')
            edge_color = EDGE_COLORS.get(edge_type, '#888')

            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=1, color=edge_color),
                hoverinfo='text',
                hovertext=f"{edge_type}: {self.short_name(u)} → {self.short_name(v)}",
                showlegend=False,
                opacity=0.6
            )
            edge_traces.append(edge_trace)

            # Arrow heads (using scatter markers)
            # Calculate arrow position (80% along the edge)
            arrow_x = x0 + 0.8 * (x1 - x0)
            arrow_y = y0 + 0.8 * (y1 - y0)

            arrow_trace = go.Scatter(
                x=[arrow_x],
                y=[arrow_y],
                mode='markers',
                marker=dict(
                    size=8,
                    color=edge_color,
                    symbol='triangle-up',
                    angle=0  # Would need calculation for proper angle
                ),
                hoverinfo='skip',
                showlegend=False,
                opacity=0.6
            )
            edge_traces.append(arrow_trace)

        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        hover_text = []

        for node_id, data in subgraph.nodes(data=True):
            if node_id not in pos:
                continue

            x, y = pos[node_id]
            node_x.append(x)
            node_y.append(y)

            node_type = data.get('type', '')
            name = data.get('name', self.short_name(node_id))

            # Node label
            if node_type == "Method":
                label = name.split("(")[0] if "(" in name else name
            else:
                label = name

            node_text.append(label if show_labels else '')
            node_color.append(NODE_COLORS.get(node_type, 'lightblue'))

            # Size based on degree
            degree = subgraph.degree(node_id)
            node_size.append(max(10, min(30, degree * 2)))

            # Hover text
            hover_parts = [f"<b>{label}</b>"]
            hover_parts.append(f"Type: {node_type}")
            hover_parts.append(f"Degree: {degree}")

            if data.get('file'):
                hover_parts.append(f"File: {data['file']}")
            if data.get('startLine'):
                hover_parts.append(f"Lines: {data['startLine']}-{data.get('endLine', '')}")
            if node_type == "Method" and data.get('return_type'):
                hover_parts.append(f"Returns: {data['return_type']}")
            if data.get('annotations'):
                annotations = data['annotations']
                if isinstance(annotations, list):
                    annotations = ', '.join(annotations)
                hover_parts.append(f"Annotations: {annotations}")

            hover_text.append("<br>".join(hover_parts))

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text' if show_labels else 'markers',
            text=node_text,
            textposition="top center",
            hovertext=hover_text,
            hoverinfo='text',
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=1, color='white')
            ),
            textfont=dict(size=8),
            showlegend=False
        )

        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])

        fig.update_layout(
            title=f"Graph View ({subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges)",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#f8f9fa',
            height=800
        )

        return fig

    def create_app(self):
        """Create Dash application"""
        app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Graph Viewer", className="mb-3"),
                    html.P(f"Total: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
                ])
            ]),

            dbc.Row([
                # Sidebar controls
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Filters", className="mb-3"),

                            # Search
                            html.Label("Search:"),
                            dbc.Input(
                                id='search-input',
                                type='text',
                                placeholder='Search nodes...',
                                debounce=True,
                                className="mb-3"
                            ),

                            # Focus node
                            html.Label("Focus Node ID:"),
                            dbc.Input(
                                id='focus-input',
                                type='text',
                                placeholder='Node ID to focus on...',
                                debounce=True,
                                className="mb-2"
                            ),
                            dbc.Input(
                                id='focus-depth',
                                type='number',
                                value=2,
                                min=1,
                                max=5,
                                step=1,
                                className="mb-3"
                            ),
                            html.Small("Focus depth (1-5)", className="text-muted mb-3"),

                            # Max nodes
                            html.Label("Max Nodes:", className="mt-2"),
                            dbc.Input(
                                id='max-nodes',
                                type='number',
                                value=500,
                                min=50,
                                max=2000,
                                step=50,
                                className="mb-3"
                            ),

                            # External nodes
                            dbc.Checkbox(
                                id='include-external',
                                label='Include External Nodes',
                                value=True,
                                className="mb-3"
                            ),

                            # Layout
                            html.Label("Layout:"),
                            dbc.Select(
                                id='layout-select',
                                options=[
                                    {'label': 'Spring (Force-directed)', 'value': 'spring'},
                                    {'label': 'Kamada-Kawai', 'value': 'kamada_kawai'},
                                    {'label': 'Circular', 'value': 'circular'},
                                    {'label': 'Hierarchical', 'value': 'hierarchical'},
                                ],
                                value='spring',
                                className="mb-3"
                            ),

                            # Labels
                            dbc.Checkbox(
                                id='show-labels',
                                label='Show Node Labels',
                                value=True,
                                className="mb-3"
                            ),

                            # Node types
                            html.Label("Node Types:", className="mt-3"),
                            dbc.Checklist(
                                id='node-types-filter',
                                options=[{'label': nt, 'value': nt} for nt in self.node_types],
                                value=self.node_types,
                                className="mb-3"
                            ),

                            # Edge types
                            html.Label("Edge Types:", className="mt-3"),
                            dbc.Checklist(
                                id='edge-types-filter',
                                options=[{'label': et, 'value': et} for et in self.edge_types],
                                value=self.edge_types,
                                className="mb-3"
                            ),

                            # Update button
                            dbc.Button(
                                "Update Graph",
                                id='update-button',
                                color='primary',
                                className="w-100 mt-3"
                            ),

                            # Export button
                            dbc.Button(
                                "Export Subgraph",
                                id='export-button',
                                color='secondary',
                                className="w-100 mt-2"
                            ),
                        ])
                    ])
                ], width=3),

                # Graph display
                dbc.Col([
                    dcc.Loading(
                        id="loading",
                        type="default",
                        children=[
                            dcc.Graph(
                                id='graph-display',
                                config={'displayModeBar': True, 'scrollZoom': True}
                            )
                        ]
                    ),
                    html.Div(id='export-status', className="mt-2")
                ], width=9)
            ])
        ], fluid=True)

        @app.callback(
            Output('graph-display', 'figure'),
            Input('update-button', 'n_clicks'),
            State('search-input', 'value'),
            State('focus-input', 'value'),
            State('focus-depth', 'value'),
            State('max-nodes', 'value'),
            State('include-external', 'value'),
            State('layout-select', 'value'),
            State('show-labels', 'value'),
            State('node-types-filter', 'value'),
            State('edge-types-filter', 'value'),
        )
        def update_graph(n_clicks, search, focus, depth, max_nodes, external, layout, labels, node_types, edge_types):
            # Filter graph
            subgraph = self.filter_graph(
                node_types=set(node_types) if node_types else set(),
                edge_types=set(edge_types) if edge_types else set(),
                include_external=external,
                search_query=search or "",
                focus_node=focus if focus and focus.strip() else None,
                focus_depth=depth or 2,
                max_nodes=max_nodes or 500
            )

            # Create figure
            fig = self.create_plotly_figure(
                subgraph,
                layout_type=layout or 'spring',
                show_labels=labels
            )

            return fig

        @app.callback(
            Output('export-status', 'children'),
            Input('export-button', 'n_clicks'),
            State('search-input', 'value'),
            State('focus-input', 'value'),
            State('focus-depth', 'value'),
            State('max-nodes', 'value'),
            State('include-external', 'value'),
            State('node-types-filter', 'value'),
            State('edge-types-filter', 'value'),
        )
        def export_subgraph(n_clicks, search, focus, depth, max_nodes, external, node_types, edge_types):
            if not n_clicks:
                return ""

            # Filter graph
            subgraph = self.filter_graph(
                node_types=set(node_types) if node_types else set(),
                edge_types=set(edge_types) if edge_types else set(),
                include_external=external,
                search_query=search or "",
                focus_node=focus if focus and focus.strip() else None,
                focus_depth=depth or 2,
                max_nodes=max_nodes or 500
            )

            # Export to gpickle
            output_file = "exported_subgraph.gpickle"
            with open(output_file, 'wb') as f:
                pickle.dump(subgraph, f)

            return dbc.Alert(
                f"Exported {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges to {output_file}",
                color="success",
                duration=4000
            )

        return app


def main():
    parser = argparse.ArgumentParser(description='Interactive Graph Viewer')
    parser.add_argument(
        '--graph',
        type=str,
        default='domain_graph.gpickle',
        help='Path to pickled NetworkX graph file'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8050,
        help='Port to run the server on'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host to run the server on'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode'
    )

    args = parser.parse_args()

    viewer = GraphViewer(args.graph)
    app = viewer.create_app()

    print(f"\n{'='*60}")
    print(f"Starting Graph Viewer on http://{args.host}:{args.port}")
    print(f"{'='*60}\n")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
