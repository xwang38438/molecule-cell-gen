import scipy as sp
import graph_tool as gt
import pickle
import torch
import graph_tool.topology as top
import networkx as nx
from graph_tool import Graph
from itertools import zip_longest

# extract rings from graph with the size of rings as parameter
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
    edges = list(G.edges())
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index



def incidence_matrix(graph, signed=False, index=False, rank=1):
    if rank == 0:
        A = sp.sparse.lil_matrix((0, len(graph.nodes())))
        if index:
            node_index = {node: i for i, node in enumerate(sorted(graph.nodes()))}
            if signed:
                return {}, node_index, A.asformat("csr")
            else:
                return {}, node_index, abs(A.asformat("csr"))
        else:
            if signed:
                return A.asformat("csr")
            else:
                return abs(A.asformat("csr"))
    elif rank == 1:
        nodelist = sorted(graph.nodes())  
        edgelist = sorted([sorted(e) for e in graph.edges()])
        A = sp.sparse.lil_matrix((len(nodelist), len(edgelist)))
        node_index = {node: i for i, node in enumerate(nodelist)}
        for ei, e in enumerate(edgelist):
            (u, v) = e[:2]
            ui = node_index[u]
            vi = node_index[v]
            A[ui, ei] = -1
            A[vi, ei] = 1

        if index:
            edge_index = {tuple(sorted(edge)): i for i, edge in enumerate(edgelist)}
            if signed:
                return node_index, edge_index, A.asformat("csr")
            else:
                return node_index, edge_index, abs(A.asformat("csr"))
        else:
            if signed:
                return A.asformat("csr")
            else:
                return abs(A.asformat("csr"))
    elif rank == 2:
        edgelist = sorted([sorted(e) for e in graph.edges()])
        cells = get_rings(graph_to_edge_index(graph))
        A = sp.sparse.lil_matrix((len(edgelist), len(cells)))

        edge_index = {
            tuple(sorted(edge)): i for i, edge in enumerate(edgelist)
        }  # orient edges

        for celli, cell in enumerate(cells):
            edge_visiting_dic = {} 
            boundary = list(zip_longest(list(list(cell)), list(list(cell))[1:] + [list(list(cell))[0]]))

            for edge in boundary:
                ei = edge_index[tuple(sorted(edge))]
                if ei not in edge_visiting_dic:
                    if edge in edge_index:
                        edge_visiting_dic[ei] = 1
                    else:
                        edge_visiting_dic[ei] = -1
                else:
                    if edge in edge_index:
                        edge_visiting_dic[ei] = edge_visiting_dic[ei] + 1
                    else:
                        edge_visiting_dic[ei] = edge_visiting_dic[ei] - 1

                A[ei, celli] = edge_visiting_dic[
                    ei
                ]  # this will update everytime we visit this edge for non-regular cell complexes
                # the regular case can be handled more efficiently :
                # if edge in edge_index:
                #    A[ei, celli] = 1
                # else:
                #    A[ei, celli] = -1
        if index:
            cell_index = {c: i for i, c in enumerate(cells)}
            if signed:
                return edge_index, cell_index, A.asformat("csr")
            else:
                return edge_index, cell_index, abs(A.asformat("csr"))
        else:
            if signed:
                return A.asformat("csr")
            else:
                return abs(A.asformat("csr"))
    else:
        raise ValueError(f"Only dimensions 0, 1 and 2 are supported, got {rank}.")
    

def down_laplacian_matrix(graph, signed=False, index=False, rank=1):
    #suppose 0 < rank <= self.dim
    row, column, B = incidence_matrix(graph, index=True)
    L_down = B.transpose() @ B
    if signed:
        L_down = abs(L_down)
    if index:
        return row, torch.tensor(L_down.todense(), dtype=torch.float)
    else:
        return torch.tensor(L_down.todense(), dtype=torch.float)
    
def up_laplacian_matrix(graph, signed=False, index=False, rank=1):
    if rank == 0:
        row, col, B_next = incidence_matrix(graph,
            rank= rank + 1, index=True
            )
        L_up = B_next @ B_next.transpose()
    #rank < self.dim:
    else: 
        row, col, B_next = incidence_matrix(graph,
            rank = rank + 1, index=True
        )
        L_up = B_next @ B_next.transpose()
    
    if not signed:
        L_up = abs(L_up)

    if index:
        return row, torch.tensor(L_up.todense(), dtype=torch.float)
    else:
        return torch.tensor(L_up.todense(), dtype=torch.float)
