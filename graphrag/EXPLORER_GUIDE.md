# Graph Explorer v2 - Quick Start Guide

## Running the Explorer

```bash
python graph_explorer_v2.py --graph domain_graph.gpickle
```

Open http://127.0.0.1:8050 in your browser.

## Features

### 🔍 Search (New!)
1. Type in the search box (minimum 2 characters)
2. Results appear as clickable buttons showing:
   - Node name
   - Node type (Class, Method, etc.)
3. Click any result to select that node
4. Search box clears and node is selected automatically
5. Click "Update" to view its neighborhood

### 🎯 Node Selection

**Two ways to select a node:**

**Method 1: Search**
- Type partial name in search box
- Click matching result
- Node is now selected

**Method 2: Click on Graph**
- Click any visible node in the graph
- Selected node turns gold
- Click "Update" to explore from there

### ⚙️ Exploration Controls

Once a node is selected:

1. **Direction**
   - `↔️ Both`: Show incoming AND outgoing edges
   - `→ Outgoing`: Only show what this node calls/uses/creates
   - `← Incoming`: Only show what depends on/calls this node

2. **Depth** (1-5 hops)
   - 1: Direct neighbors only
   - 2: Neighbors + their neighbors (default)
   - 3-5: Wider neighborhood

3. **Layout**
   - Spring: Force-directed (good for clusters)
   - Kamada-Kawai: Minimizes edge lengths
   - Circular: Nodes in a circle

4. **Display Options**
   - Show labels: Toggle node text labels
   - Include external: Show/hide external dependencies

5. **Filters**
   - Node Types: Select which types to show
   - Edge Types: Select which relationships to show

### 🔄 Update Button

**Important**: After selecting a node or changing settings, click **"🔄 Update"** to refresh the view.

### 🏠 Reset Button

Click **"🏠 Reset"** to go back to the full graph view (limited to top 500 nodes).

## Example Workflows

### Find and Explore a Specific Method

1. Type method name in search (e.g., "createOrder")
2. Click the matching result
3. Select "→ Outgoing" to see what it calls
4. Set depth to 2
5. Click "🔄 Update"
6. You now see the 2-hop call chain from that method

### Find Who Calls a Method

1. Search for the method
2. Click the result
3. Select "← Incoming"
4. Set depth to 3
5. Uncheck "Include external" (to see only internal callers)
6. Click "🔄 Update"

### Explore a Class's Dependencies

1. Search for the class name
2. Click the result
3. Select "→ Outgoing"
4. In Edge Types, uncheck everything except "calls" and "creates"
5. Set depth to 2
6. Click "🔄 Update"

### Trace Event Publishing

1. Search for an event class (e.g., "OrderCreatedEvent")
2. Click result
3. Select "← Incoming"
4. In Edge Types, select only "publishes_event"
5. Click "🔄 Update"
6. See all methods that publish this event

## Tips

- **Search is case-insensitive** - type "order" to find "OrderService"
- **Partial matching** - "create" will match "CreateOrderCommand", "createOrder", etc.
- **Search by type** - type "entity" to find all Entity nodes
- **Hover over nodes** - See detailed info without selecting
- **Zoom and pan** - Use mouse wheel to zoom, drag to pan
- **Degree-based sizing** - Larger nodes have more connections
- **Color coding** - Each node type has a distinct color (see legend in code)

## Troubleshooting

**Search returns no results**
- Check spelling
- Try shorter query (e.g., "Order" instead of "OrderService")
- Try searching by type (e.g., "Class", "Method")

**Node click doesn't work**
- Look for terminal output: "✓ Node clicked: <id>"
- If no output, the click event isn't registering (report this as a bug)
- Try clicking the node marker (circle), not the label text

**Graph is empty after clicking Update**
- Check filters - you may have excluded all relevant nodes/edges
- Try "🏠 Reset" to start over
- Increase depth if neighborhood is very sparse

**Layout is messy**
- Try different layout algorithms
- Reduce depth to see fewer nodes
- Filter by specific edge types to reduce clutter

## Performance Notes

- Full graph view limited to 500 highest-degree nodes
- Search returns max 20 results
- Larger depths (4-5) may take longer to render
- Kamada-Kawai layout can be slow for >100 nodes
- Use Spring layout for best performance

## Console Output

Watch the terminal for debug messages:
- `✓ Node clicked: <node_id>` - Graph click successful
- `✓ Node selected from search: <node_id>` - Search selection successful
- `✓ Reset to full graph` - Reset button clicked
- `Callback triggered by: <trigger>` - Shows what caused the update
