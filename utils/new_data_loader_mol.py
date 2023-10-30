import os
from time import time
import numpy as np
import networkx as nx
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import json
from data.preprocess_for_nspdk import smiles_to_mols, mols_to_nx

### Code adapted from GraphEBM
def load_mol(filepath):
    print(f'Loading file {filepath}')
    if not os.path.exists(filepath):
        raise ValueError(f'Invalid filepath {filepath} for dataset')
    load_data = np.load(filepath)
    result = []
    i = 0
    while True:
        key = f'arr_{i}'
        if key in load_data.keys():
            result.append(load_data[key])
            i += 1
        else:
            break
    return list(map(lambda x, a: (x, a), result[0], result[1]))


class MolDataset(Dataset):
    def __init__(self, mols, transform):
        self.mols = mols
        self.transform = transform

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, idx):
        return self.transform(self.mols[idx])


def get_transform_fn(dataset):
    if dataset == 'QM9':
        def transform(data):
            x = 

    elif dataset == 'ZINC250k':
        def transform(data):
            

    return transform

# to do: add other adjacency matrices

def dataloader(config, get_graph_list=False):
    start_time = time()
    
    mols = load_mol(os.path.join(config.data.dir, f'{config.data.data.lower()}_kekulized.npz'))

    with open(os.path.join(config.data.dir, f'valid_idx_{config.data.data.lower()}.json')) as f:
        test_idx = json.load(f)
        
    if config.data.data == 'QM9':
        test_idx = test_idx['valid_idxs']
        test_idx = [int(i) for i in test_idx]
        col = 'SMILES1'
    
    train_idx = [i for i in range(len(mols)) if i not in test_idx]

    smiles = pd.read_csv(f'data/{config.data.data.lower()}.csv')[col]
    train_mols = [mols[i] for i in train_idx]
    test_mols = [mols[i] for i in test_idx]

    train_smiles = [smiles.iloc[i] for i in train_idx]
    test_smiles = [smiles.iloc[i] for i in test_idx]
    train_nx_graphs = mols_to_nx(smiles_to_mols(train_smiles))
    test_nx_graphs = mols_to_nx(smiles_to_mols(test_smiles))

    # apply transform with topomodelx functions


    # print(f'Number of training mols: {len(train_idx)} | Number of test mols: {len(test_idx)}')

    # train_mols = [mols[i] for i in train_idx]
    # test_mols = [mols[i] for i in test_idx]

    # train_dataset = MolDataset(train_mols, get_transform_fn(config.data.data))
    # test_dataset = MolDataset(test_mols, get_transform_fn(config.data.data))

    # if get_graph_list:
    #     train_mols_nx = [nx.from_numpy_matrix(np.array(adj)) for x, adj in train_dataset]
    #     test_mols_nx = [nx.from_numpy_matrix(np.array(adj)) for x, adj in test_dataset]
    #     return train_mols_nx, test_mols_nx

    # train_dataloader = DataLoader(train_dataset, batch_size=config.data.batch_size, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=config.data.batch_size, shuffle=True)

    # print(f'{time() - start_time:.2f} sec elapsed for data loading')
    # return train_dataloader, test_dataloader