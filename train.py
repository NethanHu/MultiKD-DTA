import sys
import os

import torch
import torch.nn as nn
import numpy as np
from rich.console import Console
from rich import print
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from torch.nn.utils.rnn import pad_sequence

from datetime import datetime as dt

console = Console()

from src.utils import *
from src.statistics import *
from src.ISBRA import Predictor

# Apply the APEX package developed by NVIDIA
from apex import amp

dataset = ['davis', 'kiba'][0]
modeling = Predictor
model_st = modeling.__name__

cuda_name = "cuda:0"

TRAIN_BATCH_SIZE = 200
TEST_BATCH_SIZE = 200
LR = 1e-3
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 500

# Restore from checkpoint?
RESTORE = False

loss_fn = nn.MSELoss()

device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")

processed_train_file_path = 'data/processed/' + dataset + '_train.pt'
processed_test_file_path = 'data/processed/' + dataset + '_test.pt'

console.log('Loading ESM-2 (Meta protein big model) from disk...')
if dataset == 'davis':
    z = np.load('./davis.npz', allow_pickle=True)
else:
    z = np.load('./.npz', allow_pickle=True)
z = z['dict'][()]


def train(model, device, train_loader, loss_fn, optimizer):
    model.train()
    for data in tqdm(train_loader, desc='Train Process'):
        '''
        To temporarily compute protein/target data, although it may slow down the training speed, 
        is to trade off time for space. The purpose here is to truncate or pad zeros to achieve a fixed length.
        '''
        target = [torch.from_numpy(z[data.pid[i].item()].squeeze())[:1000, :] for i in range(TRAIN_BATCH_SIZE)]
        target = pad_sequence(target).permute(1, 0, 2)
        target = target.to(device)
        data = data.to(device)
        '''
        data.target => Size{list: batch-size, Tensor(1000, 640)}
        Using a large protein model here would lead to significant memory consumption.
        '''
        optimizer.zero_grad()
        _, out = model(data, target)
        loss = loss_fn(out, data.y.reshape(-1).float().to(device))
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        # loss.backward()
        optimizer.step()


def predict(model, device, test_loader):
    total_preds, total_labels = torch.Tensor(), torch.Tensor()
    with torch.no_grad():
        for data in tqdm(test_loader, desc='Test  Process'):
            target = [torch.from_numpy(z[data.pid[i].item()].squeeze())[:1000, :] for i in range(TRAIN_BATCH_SIZE)]
            target = pad_sequence(target).permute(1, 0, 2)
            target = target.to(device)
            data = data.to(device)
            _, out = model(data, target)
            total_preds = torch.cat((total_preds, out.to('cpu')), dim=0)
            total_labels = torch.cat((total_labels, data.y.reshape(-1, 1).to('cpu')), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


console.log('Starting running the train program...')
console.log('[Train Args] Dataset: {}'.format(dataset))
console.log('[Train Args] Model: {}'.format(model_st))
console.log('[Train Args] Learning Rate: {}'.format(LR))
console.log('[Train Args] Epochs: {}'.format(NUM_EPOCHS))

if not os.path.exists(processed_train_file_path) or not os.path.exists(processed_test_file_path):
    console.log('[Warning] Processed training files do not exist!')
    console.log('[Warning] You need to run the generate_drug_profile.py first!')
    sys.exit(0)

console.log('[System Info] Loading training data from disk, it takes a while...')

train_drug, train_protein, train_affinity, pid = getdata_from_csv('./data/' + dataset + '_train.csv', maxlen=1536)

train_affinity = torch.from_numpy(np.array(train_affinity)).float()

train_data = TestbedDataset(root='data', dataset=dataset + '_train', pid=pid)
test_data = TestbedDataset(root='data', dataset=dataset + '_test', pid=pid)

train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, drop_last=True)

model = modeling().to(device)
if RESTORE:
    model.load_state_dict(torch.load('model/{}_model.pt'.format(dataset)))
    console.log('[Train Args] Find the previous model file!')
    console.log('[Train Args] Restoring checkpoint from the file...')

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")  # Convert FP32 to FP16
best_mse = 1000
best_ci = 0
best_auroc = 0
best_aupr = 0

model_file_name = 'model_' + model_st + '_' + dataset + '.model'
metrics_file_name = 'model_' + model_st + '_' + dataset + '.csv'

for epoch in range(NUM_EPOCHS):
    tik = dt.now()
    console.log('--> Epoch: {}/{}, Dataset: {}, Model: {}'.format(epoch + 1, NUM_EPOCHS, dataset, model_st))
    train(model, device, train_loader, loss_fn, optimizer)
    G, P = predict(model, device, test_loader)
    metrics = {'rmse': rmse(G, P), 'mse': mse(G, P), 'pearson': pearson(G, P), 'spearman': spearman(G, P),
               'ci': optimized_ci(G, P)}
    if metrics['mse'] < best_mse:
        best_mse = metrics['mse']
    if metrics['ci'] > best_ci:
        best_ci = metrics['ci']
        console.log('CI has been improved... Saving model to file...')
        if os.path.exists('model/') == False:
            os.mkdir('model')
        torch.save(model.state_dict(), 'model/{}_model.pt'.format(dataset))
    tok = dt.now()
    print('Training time of this epoch -> {}s'.format((tok - tik).seconds))
    print('Current Metrics -> MSE:{:.3f}, CI:{:.3f}'.format(
        metrics['mse'], metrics['ci']
    ))
    print('Best    Metrics -> MSE:{:.3f}, CI:{:.3f}'.format(
        best_mse, best_ci
    ))
