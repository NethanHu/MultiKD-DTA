import os
import numpy as np
import pandas as pd


import torch
from torch_geometric.data import InMemoryDataset, Data

from tqdm import tqdm
from rich.console import Console

console = Console()


class Alphabets:
    def __init__(self, chars, encoding=None, missing=255):
        self.chars = np.frombuffer(chars, dtype='uint8')
        self.size = len(self.chars)
        self.encoding = np.zeros(256, dtype='uint8') + missing
        if encoding == None:
            self.encoding[self.chars] = np.arange(self.size)
        else:
            self.encoding[self.chars] = encoding

    def encode(self, s):
        s = np.frombuffer(s, dtype='uint8')
        return self.encoding[s]


class Smiles(Alphabets):
    def __init__(self):
        chars = b'#%)(+-.1032547698=ACBEDGFIHKMLONPSRUTWVY[Z]_acbedgfihmlonsruty'
        super(Smiles, self).__init__(chars)


def getdata_from_csv(fname, maxlen=512):
    df = pd.read_csv(fname)
    smiles = list(df['compound_iso_smiles'])
    protein = list(df['target_sequence'])
    affinity = list(df['affinity'])
    pid = list(df['protein_id'])
    return smiles, protein, affinity, pid


class TestbedDataset(InMemoryDataset):
    def __init__(self, root, dataset, drug=None, protein=None, affinity=None, pid=None, smile_graph=None,
                 transform=None, pre_transform=None):
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.drug = drug
        self.protein = protein
        self.affinity = affinity
        self.pid = pid
        self.smile_graph = smile_graph
        if os.path.isfile(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.data, self.slices = self.process()

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self):
        assert (len(self.drug) == len(self.protein)
                and len(self.drug) == len(self.affinity)), 'Sizes of Drug, Protein and Affinity should be the same!'

        # 对 SMILES Graph 和 Protein 引入到 PyG 的 dataloader 操作
        gnn_db = []
        gnn_db_size = len(self.affinity)
        for i in tqdm(range(gnn_db_size), unit='it', desc='Converting process'):
            smile, target, label = self.drug[i], self.protein[i], self.affinity[i]
            c_size, features, edge_index = self.smile_graph[smile]
            gnn_data = Data(x=torch.Tensor(np.array(features)),
                            edge_index=torch.LongTensor(np.array(edge_index)).transpose(1, 0),
                            y=torch.FloatTensor(np.array([label])))
            gnn_data.target = None  # 这里我们先不进行处理，在下面的 getitem 方法处进行处理
            gnn_data.c_size = torch.Tensor(np.array([c_size]))
            gnn_data.pid = self.pid[i]
            gnn_db.append(gnn_data)

        if self.pre_filter is not None:
            gnn_db = [gnn_data for gnn_data in gnn_db if self.pre_filter(gnn_data)]

        if self.pre_transform is not None:
            gnn_db = [self.pre_transform(gnn_data) for gnn_data in gnn_db]

        console.log('Graph construction complete! Saving to file...')
        data, slices = self.collate(gnn_db)
        torch.save((data, slices), self.processed_paths[0])
        return data, slices