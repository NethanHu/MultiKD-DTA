import os
import json, pickle
from collections import OrderedDict

import pandas as pd
import numpy as np

from rdkit.Chem import MolFromSmiles, MolToSmiles
import networkx as nx

from tqdm import tqdm
from rich.console import Console

from src.utils import TestbedDataset

console = Console()

seq_dict = {v: (i + 1) for i, v in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}  # 仅使用氨基酸中使用的20个字母
seq_dict_len = len(seq_dict)
max_seq_len = 1000
datasets = ['kiba']
compound_iso_smiles = []
smile_graph_dict = {}


# DeepDTA 的编码操作
def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(), ['H', 'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl',
                                                             'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V',
                                                             'K',
                                                             'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                                                             'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                                             'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


# smile_to_graph 函数是用来生成 features, edge_index 的函数，如果要改进输入的格式就在这里操作
# data.x: Node feature matrix with shape [num_nodes, num_node_features]
# data.edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
# data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
# data.y: Target to train against (may have arbitrary shape), e.g., node-level targets of shape [num_nodes, *]
#         or graph-level targets of shape [1, *]
# data.pos: Node position matrix with shape [num_nodes, num_dimensions]
def smile_to_graph(smile):
    mol = MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    features, edges, edge_index = [], [], []

    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    g = nx.Graph(edges).to_directed()
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index


# 这部分是将 proteins 规定为 1000 的序列，多删少补（0）
def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x


for dataset in datasets:
    console.log('Ready to convert data in {}...'.format(dataset))
    file_path = 'data/' + dataset + '/'
    # 读入 train fold，然后完成 Flatten 的操作
    train_fold = [__ for _ in json.load(open(file_path + 'folds/train_fold_setting1.txt')) for __ in _]
    valid_fold = json.load(open(file_path + 'folds/test_fold_setting1.txt'))
    ligands = json.load(open(file_path + 'ligands_can.txt'), object_pairs_hook=OrderedDict)
    proteins = json.load(open(file_path + 'proteins.txt'), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(file_path + 'Y', 'rb'), encoding='latin1')
    console.log('[Load Source File] Load file from disk completed, next...')
    console.log('[Analysis 3D Structure] Analysis 3D structures of SMILES, converting...')
    drugs, prots = [], []
    for d in ligands.keys():
        lg = MolToSmiles(MolFromSmiles(ligands[d]), isomericSmiles=True)
        drugs.append(lg)
    for p in proteins.keys():
        prots.append(proteins[p])

    if dataset == 'davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)

    for opt in ['train', 'test']:
        console.log('[Data Cleaning] Initially filter the {} data in {}...'.format(opt, dataset))
        rows, cols = np.where(np.isnan(affinity) == False)
        if opt == 'train':
            rows, cols = rows[train_fold], cols[train_fold]
        else:
            rows, cols = rows[valid_fold], cols[valid_fold]

        with open('data/' + dataset + '_' + opt + '.csv', 'w') as f:
            f.write('compound_iso_smiles,target_sequence,affinity\n')
            for i in range(len(rows)):
                line = [drugs[rows[i]], prots[cols[i]], affinity[rows[i], cols[i]]]
                f.write(','.join(map(str, line)) + '\n')

        console.log('[Summary] Filter finished, here is the summary in brief:')
        console.log('[Summary] Total train fold size: {}, test fold size: {}'.format(len(train_fold), len(valid_fold)))

console.log('[SMILES to Graph] Converting SMILES to graph data structure')
for dataset in datasets:
    for opt in ['train', 'test']:
        df = pd.read_csv('data/' + dataset + '_' + opt + '.csv')
        compound_iso_smiles += list(df['compound_iso_smiles'])
# 精髓是这里的 set，去重
console.log('Before deleting the duplicated data, dataset size: {}'.format(len(compound_iso_smiles)))
compound_iso_smiles = set(compound_iso_smiles)  # 140K -> 2K
console.log('After deleting the duplicated data, dataset size: {}'.format(len(compound_iso_smiles)))
for smile in compound_iso_smiles:
    smile_graph_dict[smile] = smile_to_graph(smile)

console.log('[Prepare Input Data] Starting collect necessary train and test data')
for dataset in datasets:
    pretrained_file_path = 'data/' + dataset + '/' + dataset + '.npz'
    processed_train_path = 'data/processed/' + dataset + '_train.pt'
    processed_test_path = 'data/processed/' + dataset + '_test.pt'
    if not os.path.isfile(processed_train_path) or not os.path.isfile(processed_test_path):
        console.log('[Prepare Input Data] Starting preparing data from graph to pytorch format.')
        for opt in ['train', 'test']:
            df = pd.read_csv('data/' + dataset + '_' + opt + '.csv')
            drugs, prots, Y, pid = (list(df['compound_iso_smiles']), list(df['target_sequence']), list(df['affinity']),
                                    list(df['protein_id']))
            # XT = [seq_cat(p) for p in prots]  # XT是 1000 长度的 protein 由字母映射成具体数字之后的序列
            drugs, prots, Y = np.asarray(drugs), np.asarray(prots), np.asarray(Y)
            if opt == 'train':
                train_data = TestbedDataset(root='data', dataset=dataset + '_train', drug=drugs, protein=prots,
                                            affinity=Y, pid=pid, smile_graph=smile_graph_dict)
            else:
                test_data = TestbedDataset(root='data', dataset=dataset + '_test', drug=drugs, protein=prots,
                                           affinity=Y, pid=pid, smile_graph=smile_graph_dict)
    else:
        console.log('[Finish] Already find train_data and test_data, skip this process...')
