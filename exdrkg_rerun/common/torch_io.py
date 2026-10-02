"""Loading helpers for PyG HeteroData files."""
import torch


def register_pyg_safe_globals():
    """Allow torch.load(weights_only=True) to unpickle PyG graph containers; harmless when weights_only=False is used."""
    from torch_geometric.data import HeteroData
    from torch_geometric.data.storage import BaseStorage, NodeStorage, EdgeStorage, GlobalStorage
    torch.serialization.add_safe_globals([HeteroData, BaseStorage, NodeStorage, EdgeStorage, GlobalStorage])


def load_kg(path, map_location="cpu"):
    register_pyg_safe_globals()
    return torch.load(path, weights_only=False, map_location=map_location)
