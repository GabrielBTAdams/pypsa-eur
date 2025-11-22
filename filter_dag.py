#!/usr/bin/env python3
"""
Filter Snakemake DAG files to show selected nodes and their upstream dependencies.

Usage:
    python filter_dag.py input.dot output.dot --nodes node1 node2 --depth 2
"""

import re
import argparse
from collections import defaultdict, deque
from typing import Set, Dict, List, Tuple


def parse_dot_file(filepath: str) -> Tuple[Dict, List[Tuple[str, str]], str]:
    """
    Parse a .dot file and extract nodes, edges, and graph attributes.
    
    Returns:
        nodes: dict mapping node_id to node definition
        edges: list of (from_id, to_id) tuples
        graph_type: 'rulegraph' or 'filegraph'
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Determine graph type
    graph_type = 'filegraph' if 'filegraph' in filepath or 'shape=none' in content else 'rulegraph'
    
    nodes = {}
    edges = []
    
    # Parse nodes - handle both simple and HTML-table formats
    # Simple format: 0[label = "all", color = "0.01 0.6 0.85", style="rounded"];
    simple_node_pattern = r'(\d+)\[label\s*=\s*"([^"]+)"[^\]]*\];?'
    
    # HTML table format for filegraph
    html_node_pattern = r'(\d+)\s*\[\s*shape=none.*?</table>>]'
    
    for match in re.finditer(simple_node_pattern, content):
        node_id = match.group(1)
        nodes[node_id] = match.group(0)
    
    for match in re.finditer(html_node_pattern, content, re.DOTALL):
        node_id = match.group(1)
        nodes[node_id] = match.group(0)
    
    # Parse edges: from -> to
    edge_pattern = r'(\d+)\s*->\s*(\d+)'
    for match in re.finditer(edge_pattern, content):
        from_id = match.group(1)
        to_id = match.group(2)
        edges.append((from_id, to_id))
    
    return nodes, edges, graph_type


def extract_node_label(node_def: str) -> str:
    """Extract the label/name from a node definition."""
    # For simple nodes
    simple_match = re.search(r'label\s*=\s*"([^"]+)"', node_def)
    if simple_match:
        return simple_match.group(1)
    
    # For HTML table nodes
    html_match = re.search(r'<b><font point-size="18">([^<]+)</font></b>', node_def)
    if html_match:
        return html_match.group(1)
    
    return "unknown"


def build_graph(edges: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """Build adjacency list for upstream navigation (reverse direction)."""
    graph = defaultdict(list)
    for from_id, to_id in edges:
        # Reverse: to find upstream, we need to know what feeds into each node
        graph[to_id].append(from_id)
    return graph


def find_upstream_nodes(
    target_nodes: Set[str],
    graph: Dict[str, List[str]],
    max_depth: int
) -> Set[str]:
    """
    Find all upstream nodes within max_depth hops from target nodes.
    Uses BFS to traverse upstream dependencies.
    """
    visited = set(target_nodes)
    queue = deque([(node, 0) for node in target_nodes])
    
    while queue:
        current, depth = queue.popleft()
        
        if depth < max_depth:
            # Get upstream nodes (nodes that feed into current)
            for upstream in graph.get(current, []):
                if upstream not in visited:
                    visited.add(upstream)
                    queue.append((upstream, depth + 1))
    
    return visited


def filter_dag(
    input_file: str,
    output_file: str,
    target_labels: List[str],
    max_depth: int = 2
):
    """
    Filter DAG file to include only target nodes and their upstream dependencies.
    """
    # Parse input file
    nodes, edges, graph_type = parse_dot_file(input_file)
    
    # Create label to ID mapping
    label_to_ids = defaultdict(list)
    for node_id, node_def in nodes.items():
        label = extract_node_label(node_def)
        label_to_ids[label].append(node_id)
    
    # Find target node IDs
    target_ids = set()
    for label in target_labels:
        if label in label_to_ids:
            target_ids.update(label_to_ids[label])
        else:
            print(f"Warning: Node '{label}' not found in graph")
    
    if not target_ids:
        print("Error: No valid target nodes found")
        return
    
    # Build upstream graph
    graph = build_graph(edges)
    
    # Find all nodes to include
    nodes_to_include = find_upstream_nodes(target_ids, graph, max_depth)
    
    # Filter edges
    filtered_edges = [
        (from_id, to_id) 
        for from_id, to_id in edges 
        if from_id in nodes_to_include and to_id in nodes_to_include
    ]
    
    # Write output
    write_dot_file(output_file, nodes, nodes_to_include, filtered_edges, graph_type)
    
    # Print statistics
    print(f"Filtered graph created: {output_file}")
    print(f"  Original nodes: {len(nodes)}")
    print(f"  Filtered nodes: {len(nodes_to_include)}")
    print(f"  Original edges: {len(edges)}")
    print(f"  Filtered edges: {len(filtered_edges)}")


def write_dot_file(
    filepath: str,
    nodes: Dict,
    included_ids: Set[str],
    edges: List[Tuple[str, str]],
    graph_type: str
):
    """Write filtered graph to .dot file."""
    with open(filepath, 'w') as f:
        # Write header
        f.write("digraph snakemake_dag {\n")
        f.write("    graph[bgcolor=white, margin=0];\n")
        f.write("    node[shape=box, style=rounded, fontname=sans,                 fontsize=10, penwidth=2];\n")
        f.write("    edge[penwidth=2, color=grey];\n")
        
        # Write nodes
        for node_id in sorted(included_ids, key=int):
            if node_id in nodes:
                f.write(nodes[node_id])
                f.write("\n")
        
        # Write edges
        for from_id, to_id in edges:
            f.write(f"\t{from_id} -> {to_id}\n")
        
        f.write("}\n")


def export_to_pdf(dot_file: str, pdf_file: str):
    """Export .dot file to PDF using Graphviz."""
    import subprocess
    
    try:
        subprocess.run(
            ["dot", "-Tpdf", dot_file, "-o", pdf_file],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"PDF created: {pdf_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating PDF: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: Graphviz 'dot' command not found. Please install Graphviz.")
        print("  Ubuntu/Debian: sudo apt-get install graphviz")
        print("  macOS: brew install graphviz")
        print("  Windows: Download from https://graphviz.org/download/")
        return False


def main():
    # ========== CONFIGURATION ==========
    # Input file
    input_file = "doc/img/rulegraph-tutorial-overnight.dot"
    input_file = "doc/img/filegraph-tutorial-overnight.dot"
    
    # Output files (PDF will have same name with .pdf extension)
    output_dot = "filtered-rulegraph.dot"
    
    
    # Target nodes to filter
    # target_nodes = ["build_transport_demand", "prepare_sector_network"]
    target_nodes = ["build_transport_demand"]
    
    # Maximum depth of upstream dependencies
    max_depth = 1
    
    # Export to PDF (requires Graphviz installed)
    export_pdf = True
    # ===================================
    
    # Filter the DAG
    filter_dag(input_file, output_dot, target_nodes, max_depth)
    
    # Export to PDF if requested
    if export_pdf:
        pdf_file = output_dot.rsplit(".", 1)[0] + ".pdf"
        export_to_pdf(output_dot, pdf_file)


if __name__ == "__main__":
    main()