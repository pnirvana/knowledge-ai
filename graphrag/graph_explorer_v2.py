#!/usr/bin/env python3
"""
Interactive Graph Explorer - Simplified with Working Click Events

Usage:
    python graph_explorer_v2.py [--graph domain_graph.gpickle] [--port 8050]
"""

import argparse
import pickle
import networkx as nx
import plotly.graph_objs as go
from dash import Dash, html, dcc, Input, Output, State, callback_context, ALL
import dash_bootstrap_components as dbc
from typing import Set, Dict, Tuple
import json

# Color palettes
NODE_COLORS = {
    "UseCase": "orange", "Entity": "red", "Class": "steelblue",
    "Interface": "lightsteelblue", "Enum": "lightslategray",
    "Event": "green", "Attribute": "lightgreen", "Method": "purple",
    "ExternalClass": "#bdbdbd", "ExternalMethod": "#bdbdbd",
}

EDGE_COLORS = {
    "calls": "#2ca02c", "creates": "#ff99c8", "publishes_event": "#d62728",
    "has_attribute": "#ff7f0e", "defines": "#9e9e9e", "reads_attribute": "#1f77b4",
    "writes_attribute": "#8c564b", "implemented_by": "#9467bd", "represented_by": "#17becf",
}

EXTERNAL_PREFIX = "ext://"


class GraphExplorer:
    def __init__(self, graph_path: str):
        print(f"Loading graph from {graph_path}...")
        with open(graph_path, 'rb') as f:
            self.G = pickle.load(f)

        print(f"Loaded: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")

        self.node_types = sorted(set(d.get('type', '') for _, d in self.G.nodes(data=True) if d.get('type')))
        self.edge_types = sorted(set(d.get('type', '') for _, _, d in self.G.edges(data=True) if d.get('type')))

        print(f"Node types: {', '.join(self.node_types)}")
        print(f"Edge types: {', '.join(self.edge_types)}")

    def is_external(self, node_id: str) -> bool:
        return isinstance(node_id, str) and node_id.startswith(EXTERNAL_PREFIX)

    def short_name(self, node_id: str) -> str:
        try:
            base = node_id.split("#")[0] if "#" in node_id else node_id
            base = base[:base.index("(")] if "(" in base else base
            return base.rsplit(".", 1)[-1]
        except:
            return node_id

    def get_neighborhood(self, node_id: str, depth: int, direction: str,
                        node_types: Set[str], edge_types: Set[str],
                        include_external: bool) -> nx.DiGraph:
        """Extract neighborhood subgraph"""
        if node_id not in self.G:
            return nx.DiGraph()

        nodes = {node_id}
        current = {node_id}

        for _ in range(depth):
            next_level = set()

            for n in current:
                if direction in ('outgoing', 'both'):
                    for _, target, edata in self.G.out_edges(n, data=True):
                        if edata.get('type', '') not in edge_types:
                            continue
                        tdata = self.G.nodes[target]
                        if tdata.get('type', '') not in node_types:
                            continue
                        if not include_external and self.is_external(target):
                            continue
                        nodes.add(target)
                        next_level.add(target)

                if direction in ('incoming', 'both'):
                    for source, _, edata in self.G.in_edges(n, data=True):
                        if edata.get('type', '') not in edge_types:
                            continue
                        sdata = self.G.nodes[source]
                        if sdata.get('type', '') not in node_types:
                            continue
                        if not include_external and self.is_external(source):
                            continue
                        nodes.add(source)
                        next_level.add(source)

            current = next_level

        subgraph = self.G.subgraph(nodes).copy()

        # Filter edges
        edges_to_remove = [(u, v) for u, v, d in subgraph.edges(data=True)
                          if d.get('type', '') not in edge_types]
        subgraph.remove_edges_from(edges_to_remove)

        return subgraph

    def create_figure(self, subgraph: nx.DiGraph, layout: str,
                     show_labels: bool, selected_node: str = None) -> go.Figure:
        """Create Plotly figure"""

        if subgraph.number_of_nodes() == 0:
            return go.Figure().update_layout(
                title="Empty graph - click Reset or select a node",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )

        # Compute layout
        try:
            if layout == "spring":
                pos = nx.spring_layout(subgraph, seed=42, k=1.5, iterations=50)
            elif layout == "kamada_kawai":
                pos = nx.kamada_kawai_layout(subgraph)
            elif layout == "circular":
                pos = nx.circular_layout(subgraph)
            else:
                pos = nx.spring_layout(subgraph, seed=42)
        except:
            import random
            pos = {n: (random.random(), random.random()) for n in subgraph.nodes()}

        # Create edge traces
        edge_traces = []
        for u, v, edata in subgraph.edges(data=True):
            if u not in pos or v not in pos:
                continue

            x0, y0 = pos[u]
            x1, y1 = pos[v]
            etype = edata.get('type', '')

            edge_traces.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode='lines',
                line=dict(width=1.5, color=EDGE_COLORS.get(etype, '#888')),
                hovertext=f"{etype}: {self.short_name(u)} → {self.short_name(v)}",
                hoverinfo='text',
                showlegend=False,
                opacity=0.6
            ))

        # Create node trace
        node_data = {
            'x': [], 'y': [], 'text': [], 'color': [], 'size': [],
            'hover': [], 'ids': []
        }

        for node_id, ndata in subgraph.nodes(data=True):
            if node_id not in pos:
                continue

            x, y = pos[node_id]
            ntype = ndata.get('type', '')
            name = ndata.get('name', self.short_name(node_id))
            label = name.split("(")[0] if "(" in name else name

            node_data['x'].append(x)
            node_data['y'].append(y)
            node_data['ids'].append(node_id)
            node_data['text'].append(label if show_labels else '')

            # Highlight selected
            if node_id == selected_node:
                node_data['color'].append('gold')
                node_data['size'].append(30)
            else:
                node_data['color'].append(NODE_COLORS.get(ntype, 'lightblue'))
                node_data['size'].append(max(12, min(25, subgraph.degree(node_id) * 3)))

            # Hover text
            in_deg = subgraph.in_degree(node_id)
            out_deg = subgraph.out_degree(node_id)
            hover = f"<b>{label}</b><br>Type: {ntype}<br>In: {in_deg}, Out: {out_deg}"
            if ndata.get('file'):
                hover += f"<br>File: {ndata['file']}"
            hover += "<br><br><i>👆 Click to explore</i>"
            node_data['hover'].append(hover)

        node_trace = go.Scatter(
            x=node_data['x'],
            y=node_data['y'],
            mode='markers+text' if show_labels else 'markers',
            text=node_data['text'],
            textposition="top center",
            hovertext=node_data['hover'],
            hoverinfo='text',
            marker=dict(
                size=node_data['size'],
                color=node_data['color'],
                line=dict(width=2, color='white'),
                opacity=0.9
            ),
            textfont=dict(size=9),
            showlegend=False,
            customdata=node_data['ids']  # Single value per point
        )

        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(
            title=f"Graph Explorer ({subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges)",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#f8f9fa',
            height=800,
            clickmode='event+select'
        )

        return fig

    def create_app(self):
        app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        app.layout = dbc.Container([
            # Hidden stores
            dcc.Store(id='selected-node-store', data=None),

            html.H2("🔍 Graph Explorer", className="mt-3"),
            html.P(f"Total: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges",
                   className="text-muted mb-3"),

            dbc.Row([
                # Sidebar
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            # Search
                            html.H5("🔍 Search"),
                            dbc.Input(
                                id='search-input',
                                type='text',
                                placeholder='Search for a node...',
                                debounce=True,
                                className="mb-2"
                            ),
                            html.Div(id='search-results', className="mb-3", style={'maxHeight': '200px', 'overflowY': 'auto'}),

                            html.Hr(),

                            # Selected node info
                            html.H5("🎯 Selected Node"),
                            html.Div(id='node-info', className="mb-3",
                                    children=[html.P("Search or click a node", className="text-muted")]),

                            html.Hr(),

                            # Controls
                            html.H5("⚙️ Settings"),

                            html.Label("Direction:", className="fw-bold mt-2"),
                            dbc.RadioItems(
                                id='direction',
                                options=[
                                    {'label': '↔️ Both', 'value': 'both'},
                                    {'label': '→ Outgoing', 'value': 'outgoing'},
                                    {'label': '← Incoming', 'value': 'incoming'},
                                ],
                                value='both'
                            ),

                            html.Label("Depth:", className="fw-bold mt-3"),
                            dcc.Slider(
                                id='depth',
                                min=1, max=5, step=1, value=2,
                                marks={i: str(i) for i in range(1, 6)}
                            ),

                            html.Label("Layout:", className="fw-bold mt-3"),
                            dbc.Select(
                                id='layout',
                                options=[
                                    {'label': 'Spring', 'value': 'spring'},
                                    {'label': 'Kamada-Kawai', 'value': 'kamada_kawai'},
                                    {'label': 'Circular', 'value': 'circular'},
                                ],
                                value='spring'
                            ),

                            dbc.Checkbox(id='show-labels', label='Show labels', value=True, className="mt-3"),
                            dbc.Checkbox(id='include-external', label='Include external', value=False, className="mt-2"),

                            html.Hr(),

                            html.Label("Node Types:", className="fw-bold"),
                            dbc.Checklist(
                                id='node-types',
                                options=[{'label': t, 'value': t} for t in self.node_types],
                                value=self.node_types,
                                className="small"
                            ),

                            html.Label("Edge Types:", className="fw-bold mt-3"),
                            dbc.Checklist(
                                id='edge-types',
                                options=[{'label': t, 'value': t} for t in self.edge_types],
                                value=self.edge_types,
                                className="small"
                            ),

                            html.Hr(),

                            dbc.Button("🔄 Update", id='update-btn', color='primary', className="w-100 mb-2"),
                            dbc.Button("🏠 Reset", id='reset-btn', color='warning', className="w-100"),
                        ])
                    ], style={'maxHeight': '90vh', 'overflowY': 'auto'})
                ], width=3),

                # Graph
                dbc.Col([
                    dcc.Loading(
                        dcc.Graph(
                            id='graph',
                            config={'displayModeBar': True, 'scrollZoom': True},
                            style={'height': '85vh'}
                        )
                    )
                ], width=9)
            ])
        ], fluid=True)

        # Main callback
        @app.callback(
            Output('graph', 'figure'),
            Output('node-info', 'children'),
            Input('graph', 'clickData'),
            Input('update-btn', 'n_clicks'),
            Input('reset-btn', 'n_clicks'),
            Input('selected-node-store', 'data'),  # React to search selections
            State('direction', 'value'),
            State('depth', 'value'),
            State('layout', 'value'),
            State('show-labels', 'value'),
            State('include-external', 'value'),
            State('node-types', 'value'),
            State('edge-types', 'value'),
        )
        def update(click_data, update_clicks, reset_clicks, current_node,
                  direction, depth, layout, show_labels, include_external,
                  node_types, edge_types):

            ctx = callback_context
            triggered = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'initial'

            print(f"Update graph - triggered by: {triggered}, selected node: {current_node}")

            selected = current_node

            # Build subgraph
            if selected and selected in self.G:
                print(f"Building neighborhood for: {selected}")
                subgraph = self.get_neighborhood(
                    selected, depth, direction,
                    set(node_types or []), set(edge_types or []),
                    include_external
                )

                # Node info
                ndata = self.G.nodes[selected]
                ntype = ndata.get('type', '')
                name = ndata.get('name', self.short_name(selected))

                info = [
                    html.P([html.Strong("Node: "), html.Code(name, className="small")]),
                    html.P([html.Strong("Type: "), ntype]),
                    html.P([html.Strong("Total neighbors: "),
                           f"In: {self.G.in_degree(selected)}, Out: {self.G.out_degree(selected)}"]),
                ]
                if ndata.get('file'):
                    info.append(html.P([html.Strong("File: "), html.Small(ndata['file'])]))
            else:
                # Full graph (limited)
                print(f"No node selected (selected={selected}), showing full graph")
                all_nodes = list(self.G.nodes())
                if len(all_nodes) > 500:
                    # Show top degree nodes
                    top_nodes = sorted(all_nodes, key=lambda n: self.G.degree(n), reverse=True)[:500]
                    subgraph = self.G.subgraph(top_nodes).copy()
                else:
                    subgraph = self.G.copy()

                # Apply filters
                remove_nodes = []
                for n, d in subgraph.nodes(data=True):
                    if node_types and d.get('type', '') not in node_types:
                        remove_nodes.append(n)
                    elif not include_external and self.is_external(n):
                        remove_nodes.append(n)
                subgraph.remove_nodes_from(remove_nodes)

                remove_edges = [(u, v) for u, v, d in subgraph.edges(data=True)
                               if edge_types and d.get('type', '') not in edge_types]
                subgraph.remove_edges_from(remove_edges)

                info = [html.P("Click any node to explore from there", className="text-muted")]

            fig = self.create_figure(subgraph, layout, show_labels, selected)

            return fig, info

        # Separate callback to update selected node from graph clicks
        @app.callback(
            Output('selected-node-store', 'data'),
            Input('graph', 'clickData'),
            Input('reset-btn', 'n_clicks'),
            State('selected-node-store', 'data'),
            prevent_initial_call=True
        )
        def update_selected_node(click_data, reset_clicks, current):
            ctx = callback_context
            triggered = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

            if triggered == 'reset-btn':
                print("✓ Reset - clearing selection")
                return None

            if triggered == 'graph' and click_data:
                points = click_data.get('points', [])
                if points and 'customdata' in points[0]:
                    node_id = points[0]['customdata']
                    print(f"✓ Node clicked: {node_id}")
                    return node_id

            return current

        # Search callback
        @app.callback(
            Output('search-results', 'children'),
            Input('search-input', 'value'),
            prevent_initial_call=True
        )
        def search_nodes(query):
            if not query or len(query) < 2:
                return []

            query_lower = query.lower()
            matches = []

            # Collect all matches with scoring
            for node_id, ndata in self.G.nodes(data=True):
                name = ndata.get('name', self.short_name(node_id))
                ntype = ndata.get('type', '')
                name_lower = name.lower()

                score = 0

                # Exact match (highest priority)
                if query_lower == name_lower:
                    score = 1000
                # Starts with query
                elif name_lower.startswith(query_lower):
                    score = 500
                # Contains query in name
                elif query_lower in name_lower:
                    score = 100
                # Type match
                elif query_lower in ntype.lower():
                    score = 10

                if score > 0:
                    # Bonus for shorter names (more specific)
                    score += max(0, 50 - len(name))
                    # Bonus for non-external nodes
                    if not self.is_external(node_id):
                        score += 25

                    matches.append((score, node_id, name, ntype))

            if not matches:
                return [html.P("No matches found", className="text-muted small")]

            # Sort by score (descending) and limit to top 50
            matches.sort(reverse=True, key=lambda x: x[0])
            matches = matches[:50]

            # Create clickable list
            result_items = []
            result_items.append(
                html.Div(f"Found {len(matches)} result(s):", className="text-muted small mb-2")
            )

            for score, node_id, name, ntype in matches:
                label = name.split("(")[0] if "(" in name else name
                result_items.append(
                    dbc.Button(
                        [
                            html.Div(label, className="text-truncate", style={'maxWidth': '220px'}),
                            html.Small(ntype, className="text-muted")
                        ],
                        id={'type': 'search-result', 'index': node_id},
                        color='light',
                        size='sm',
                        className='w-100 mb-1 text-start',
                        style={'padding': '5px', 'textAlign': 'left'}
                    )
                )

            return result_items

        # Handle search result clicks
        @app.callback(
            Output('selected-node-store', 'data', allow_duplicate=True),
            Output('search-input', 'value'),
            Input({'type': 'search-result', 'index': ALL}, 'n_clicks'),
            State('selected-node-store', 'data'),
            prevent_initial_call=True
        )
        def select_from_search(clicks, current):
            ctx = callback_context

            # Debug: print all trigger info
            print(f"Search callback triggered: {ctx.triggered}")
            print(f"Clicks received: {clicks}")

            if not ctx.triggered:
                print("No trigger, returning current")
                return current, ""

            # Check if any button was actually clicked
            if not clicks or not any(c for c in clicks if c):
                print("No valid clicks, returning current")
                return current, ""

            # Get the clicked button's node_id from the trigger
            trigger_prop = ctx.triggered[0]['prop_id']
            print(f"Trigger prop: {trigger_prop}")

            # Extract the JSON part (before the .n_clicks)
            button_id = trigger_prop.rsplit('.', 1)[0]
            print(f"Button ID string: {button_id}")

            import json
            try:
                button_data = json.loads(button_id)
                node_id = button_data['index']
                print(f"✓ Node selected from search: {node_id}")

                # Clear search and return selected node
                return node_id, ""
            except Exception as e:
                print(f"Error parsing button data: {e}")
                return current, ""

        return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', default='domain_graph.gpickle')
    parser.add_argument('--port', type=int, default=8050)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    explorer = GraphExplorer(args.graph)
    app = explorer.create_app()

    print(f"\n{'='*60}")
    print(f"🚀 Graph Explorer: http://127.0.0.1:{args.port}")
    print(f"{'='*60}")
    print("💡 Click any node to explore its neighborhood")
    print("   Adjust depth and direction, then click 'Update'\n")

    app.run(host='127.0.0.1', port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
