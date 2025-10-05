# Graph Viewer - Robust Visualization for Large Graphs

A Python-based interactive graph visualization tool designed to handle large graphs that cause browser-based visualizers to fail.

## Features

✅ **Server-side rendering** - No browser memory issues
✅ **Progressive loading** - Focus on specific subgraphs
✅ **Multiple layouts** - Spring, Kamada-Kawai, Circular, Hierarchical
✅ **Advanced filtering** - By node type, edge type, search query
✅ **Focus mode** - View neighborhood around specific nodes
✅ **Export subgraphs** - Save filtered views as new graphs
✅ **Interactive controls** - Zoom, pan, hover for details
✅ **Performance limits** - Configurable max nodes (50-2000)

## Installation

Install required dependencies:

```bash
pip install dash dash-bootstrap-components plotly networkx
```

## Usage

### Basic Usage

```bash
python graph_viewer.py
```

This will start the server on http://127.0.0.1:8050

### Custom Graph File

```bash
python graph_viewer.py --graph my_graph.gpickle
```

### Custom Port

```bash
python graph_viewer.py --port 8080
```

### Debug Mode

```bash
python graph_viewer.py --debug
```

## Features Guide

### 1. Search
Type any text to filter nodes by name or label. Press Enter to apply.

### 2. Focus Mode
Enter a specific node ID and set depth (1-5) to view only that node's neighborhood:
- Depth 1: Direct neighbors only
- Depth 2: Neighbors and their neighbors (default)
- Depth 3+: Wider neighborhood

### 3. Max Nodes Limit
Set maximum nodes to display (50-2000). If filtered graph exceeds this, only the highest-degree nodes are kept.

### 4. Layout Algorithms

- **Spring (Force-directed)**: Good for general graphs, shows clusters
- **Kamada-Kawai**: Minimizes edge lengths, good for small-medium graphs
- **Circular**: Nodes arranged in circle, good for cyclic structures
- **Hierarchical**: Top-down layout (requires graphviz), good for DAGs

### 5. Filters

**Node Types**: Show/hide specific node types (Class, Method, Entity, etc.)

**Edge Types**: Show/hide specific relationship types (calls, creates, etc.)

**External Nodes**: Toggle visibility of external dependencies

### 6. Export Subgraph
Save the currently filtered view as `exported_subgraph.gpickle` for further analysis.

## Performance Tips

For very large graphs (1000+ nodes):

1. **Use Focus Mode**: Start with a specific node and small depth
2. **Limit Max Nodes**: Keep under 500 for smooth interaction
3. **Filter by Type**: Hide node/edge types you don't need
4. **Use Search**: Find specific nodes first, then focus on them
5. **Try Different Layouts**: Kamada-Kawai can be slow; Spring is usually fastest

## Example Workflows

### Explore a Specific Class
1. Enter class name in Search
2. Click "Update Graph"
3. Enter the class ID in Focus Node
4. Set depth to 2
5. Click "Update Graph" again

### View Only Business Logic
1. Uncheck "Include External Nodes"
2. In Node Types, select only: Entity, UseCase, Method
3. In Edge Types, select: calls, creates, publishes_event
4. Click "Update Graph"

### Export Clean Subgraph
1. Apply your filters
2. Click "Export Subgraph"
3. Use `exported_subgraph.gpickle` with other tools

## Comparison with Browser-based Viewers

| Feature | Cytoscape.js | Sigma.js | Graph Viewer (Python) |
|---------|-------------|----------|----------------------|
| Max nodes (smooth) | ~500 | ~1000 | 500-2000 (configurable) |
| Server-side filtering | ❌ | ❌ | ✅ |
| Memory usage | High (browser) | High (browser) | Low (server) |
| Layout computation | Browser | Browser | Server (faster) |
| Progressive loading | ❌ | ❌ | ✅ |
| Export capability | Limited | Limited | Full NetworkX |

## Troubleshooting

### "Layout failed" message
- Try a different layout algorithm
- Reduce Max Nodes
- Use Focus Mode with smaller depth

### Graph appears empty
- Check if filters are too restrictive
- Clear Search input
- Reset Node/Edge type filters to all checked

### Slow performance
- Reduce Max Nodes to 200-300
- Use Focus Mode instead of viewing entire graph
- Try Circular layout (fastest)
- Disable node labels for better performance

## Advanced Usage

### Programmatic Access

```python
from graph_viewer import GraphViewer

# Load graph
viewer = GraphViewer('domain_graph.gpickle')

# Filter programmatically
subgraph = viewer.filter_graph(
    node_types={'Class', 'Method'},
    edge_types={'calls'},
    focus_node='com.example.MyClass',
    focus_depth=2,
    max_nodes=300
)

# Export
import pickle
with open('filtered.gpickle', 'wb') as f:
    pickle.dump(subgraph, f)
```

### Custom Styling

Edit the NODE_COLORS and EDGE_COLORS dictionaries in `graph_viewer.py` to customize appearance.

## Requirements

- Python 3.7+
- dash >= 2.0
- dash-bootstrap-components >= 1.0
- plotly >= 5.0
- networkx >= 2.5

## License

Same as parent project.
