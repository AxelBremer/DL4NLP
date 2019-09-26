from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import datetime
import numpy as np

import pandas as pd
import math

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

    # Initialize the dataset and data loader (note the +1)
    test_dataset = IMDBDataset(train_or_test='test', seq_length=config.seq_length)
    test_data_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)

     # Initialize the model that we are going to use
    if not (config.recurrent_dropout_model):
        model = NN(test_dataset.vocab_size, config.embed_dim, config.hidden_dim, config.output_dim, config.n_layers, config.bidirectional, config.dropout, 0).to(device)
        model.load_state_dict(torch.load(config.filename))
    else: 
        model = Model(test_dataset.vocab_size, output_dim=config.output_dim).to(device)
        model.load_state_dict(torch.load(config.filename))


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

            preds = torch.zeros((100,config.batch_size,config.output_dim))
            
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
            # print('target',batch_targets.item()) 
            # print('mean', mean)
            # print('std', std)
            # print('\n')

    pd.DataFrame(d).to_csv(config.filename[:-3]+'_results.csv')
    return 'joe'

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--filename', type=str, required=True, help='filename of the state dict')
    parser.add_argument('--B', type=int, default=100, help='Number of times to run each test datapoint')
    parser.add_argument('--seq_length', type=int, default=200, help='Dimensionality of input sequence')
    parser.add_argument('--embed_dim', type=int, default=300, help='Dimensionality of the embeddings')
    parser.add_argument('--output_dim', type=int, default=2, help='Dimensionality of output sequence')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Number of hidden units in the model')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=1024, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.5, help='Learning rate for Adam')
    parser.add_argument('--dropout', type=float, default=0.5, help='Drop out rate')
    parser.add_argument('--train_epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--bidirectional', type=bool, default=True)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--recurrent_dropout_model', type=bool, default=True, help="Vanilla bidirectional LSTM or recurrent output LSTM")


    config = parser.parse_args()

    # Train the model
    test(config)