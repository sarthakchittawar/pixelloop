import time
import pickle
import blosc2
import networkx as nx

def load_compressed_graph_chunked(compressed_path: str) -> nx.Graph:
    """
    Load compressed graph - handles both single and chunked blosc2 formats.
    
    Args:
        compressed_path: Path to compressed .b2s file
        
    Returns:
        Reconstructed NetworkX graph
    """
    print(f"Loading compressed graph from: {compressed_path}")
    t_start = time.time()

    with open(compressed_path, "rb") as f:
        # Read format marker
        format_marker = f.read(6)

        if format_marker == b"SINGLE":
            # Single file format
            compressed_size = int.from_bytes(f.read(8), "little")
            compressed_data = f.read(compressed_size)
            decompressed = blosc2.decompress(compressed_data)
            graph_data = pickle.loads(decompressed)

        elif format_marker == b"CHUNKS":
            # Chunked format
            num_chunks = int.from_bytes(f.read(4), "little")
            original_size = int.from_bytes(f.read(8), "little")

            # Read chunk sizes
            chunk_sizes = []
            for _ in range(num_chunks):
                chunk_size = int.from_bytes(f.read(4), "little")
                chunk_sizes.append(chunk_size)

            # Read and decompress chunks
            decompressed_chunks = []
            for i, chunk_size in enumerate(chunk_sizes):
                compressed_chunk = f.read(chunk_size)
                decompressed_chunk = blosc2.decompress(compressed_chunk)
                decompressed_chunks.append(decompressed_chunk)

            # Reconstruct original data
            decompressed = b"".join(decompressed_chunks)
            graph_data = pickle.loads(decompressed)

        else:
            raise ValueError(f"Unknown format marker: {format_marker}")

    # Reconstruct NetworkX graph
    G = nx.Graph()

    # Reconstruct nodes
    node_ids = graph_data["node_ids"]
    node_maps = graph_data["node_maps"]
    node_coords = graph_data["node_coords"]
    node_pixels = graph_data["node_pixels"]
    node_types = graph_data["node_types"]

    for i, node_id in enumerate(node_ids):
        attrs = {
            "map": node_maps[i].tolist(),
            "coord_mast3r": node_coords[i],
            "pixel": node_pixels[i].tolist(),
            "type": str(node_types[i]),
        }
        G.add_node(node_id, **attrs)

    # Reconstruct edges
    edge_sources = graph_data["edge_sources"]
    edge_targets = graph_data["edge_targets"]
    edge_types = graph_data["edge_types"]
    edge_weights = graph_data["edge_weights"]

    for i in range(len(edge_sources)):
        attrs = {
            "edge_type": str(edge_types[i]),
            "weight": float(edge_weights[i]),
        }
        G.add_edge(edge_sources[i], edge_targets[i], **attrs)

    # Reconstruct graph attributes
    G.graph.update(graph_data["graph_attrs"])

    # Reconstruct all_path_lengths if it exists
    if "path_nodes" in graph_data and "path_distances" in graph_data:
        path_nodes = graph_data["path_nodes"]
        path_distances = graph_data["path_distances"]
        path_dict = dict(zip(path_nodes, path_distances))
        G.graph["all_path_lengths"] = {"weight": path_dict}

    print(f"Loaded graph: {G} in {time.time() - t_start:.3f}s")
    return G