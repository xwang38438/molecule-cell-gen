import pickle
import torch
import graph_tool as gt
import graph_tool.topology as top
import networkx as nx
from graph_tool import Graph
from itertools import zip_longest
from utils.cell_lifting import *
from utils.data_loader_mol import MolDataset, get_transform_fn
from torch.utils.data import DataLoader, Dataset
import os

#print(gt.__version__)

#------------------- test cell lifting -------------------
def get_rings(edge_index, max_k=7):
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.numpy()

    edge_list = edge_index.T
    graph_gt = gt.Graph(directed=False)
    graph_gt.add_edge_list(edge_list)
    
    gt.stats.remove_self_loops(graph_gt) 
    gt.stats.remove_parallel_edges(graph_gt)

    rings = set()
    sorted_rings = set()
    for k in range(3, max_k+1):
        pattern = nx.cycle_graph(k)
        pattern_edge_list = list(pattern.edges)
        pattern_gt = gt.Graph(directed=False)
        pattern_gt.add_edge_list(pattern_edge_list)
        sub_isos = top.subgraph_isomorphism(pattern_gt, graph_gt, induced=True, subgraph=True,
                                           generator=True)
        sub_iso_sets = map(lambda isomorphism: tuple(isomorphism.a), sub_isos)
        for iso in sub_iso_sets:
            if tuple(sorted(iso)) not in sorted_rings:
                rings.add(iso)
                sorted_rings.add(tuple(sorted(iso)))
    rings = list(rings)
    return rings

def graph_to_edge_index(G):
    # Get edges
    edges = list(G.edges())
    # Convert to tensor format
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


# create main
if __name__ == '__main__':
    with open('data/qm9_train_nx.pkl', 'rb') as f:
        zinc = pickle.load(f)

    G = zinc[0]
    # G_edge_index = graph_to_edge_index(G)
    # rings = get_rings(G_edge_index)
    # for i, ring in enumerate(rings):
    #     print(f'Ring {i}: {ring}')
    #     print('boundary:', list(zip_longest(list(ring), list(ring)[1:] + [list(ring)[0]])) )

    print(down_laplacian_matrix(G))
    # print(incidence_matrix(G, signed=False, index=False, rank=1).todense())
    # print(incidence_matrix(G, signed=False, index=False, rank=2).todense())
    print(up_laplacian_matrix(G))

# -------------------test data loader-------------------
#train_mols = pickle.load(open(os.path.join('data', f'qm9_train_nx.pkl'), 'rb'))
# test_mols = pickle.load(open(os.path.join('data', f'qm9_train_nx.pkl'), 'rb'))
# test_dataset = MolDataset(test_mols, get_transform_fn("QM9"))
# test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=True)

# data = iter(test_dataloader)
# first_batch = next(data)
# print(first_batch)

