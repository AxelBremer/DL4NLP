from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import datetime
import numpy as np

import pandas as pd
import math

import json

import torch
from torch import nn
from torch.utils.data import DataLoader

from imdb_dataset import IMDBDataset
from model import NN
from recurrent_dropout_lstm import Model



def test(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the device which to run the model on
    device = torch.device(device)

    with open(f'runs/{config.name}/args.txt', 'r') as f:
        config.__dict__ = json.load(f)

    # Initialize the dataset and data loader (note the +1)
    test_dataset = IMDBDataset(train_or_test='test', seq_length=config.seq_length)
    test_data_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)

     # Initialize the model that we are going to use
    if not (config.recurrent_dropout_model):
        model = NN(test_dataset.vocab_size, config.embed_dim, config.hidden_dim, config.output_dim, config.n_layers, config.bidirectional, config.dropout, 0).to(device)
        model.load_state_dict(torch.load(f'runs/{config.name}/model.pt'))
    else: 
        model = Model(test_dataset.vocab_size, output_dim=config.output_dim).to(device)
        model.load_state_dict(torch.load(f'runs/{config.name}/model.pt'))


    # Setup the loss and optimizer
    criterion = torch.nn.MSELoss().to(device)
    lowest = 100
    save = []
    epochs = 0
    num_steps = math.floor(25000/config.batch_size)

    d = {'target':[], 'mean0':[], 'std0':[], 'mean1':[], 'std1':[]}

    with torch.no_grad():
        for step, (batch_inputs, batch_targets) in enumerate(test_data_loader):

            x = batch_inputs.long().to(device)
            y_target = batch_targets.long().to(device)

            # print('\n*************************************************\n')
            # print(test_dataset.convert_to_string(x[0].tolist()))

            preds = torch.zeros((100,x.shape[0],config.output_dim))
            
            for i in range(config.B):
                preds[i,:,:] = model(x)

            # print('\n')

            if step % 1 == 0:
                print(step,'/',num_steps)

            mean = preds.mean(dim=0)
            std = preds.std(dim=0)

            d['target'].extend(batch_targets.tolist())
            d['mean0'].extend(mean[:,0].tolist())
            d['std0'].extend(std[:,0].tolist())
            d['mean1'].extend(mean[:,1].tolist())
            d['std1'].extend(std[:,1].tolist())

    pd.DataFrame(d).to_csv(f'runs/{config.name}/results.csv')
    return 'joe'

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--name', type=str, required=True, help='name of the run')

    config = parser.parse_args()

    # Train the model
    test(config)