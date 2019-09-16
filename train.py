from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import datetime
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import ImdbDataset
from lstm import LSTM

def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the model that we are going to use
    model = LSTM(config.input_dim, config.embed_dim, config.hidden_dim, config.output_dim, config.n_layers, config.bidirectional, config.dropout, 0).to(device) 
            
      
    # Initialize the dataset and data loader (note the +1)
    dataset = ImdbDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    highest = 0 
    save = []
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        t1 = time.time()

        predictions = model(text, text_length).to(device)

        loss = criterion(predictions, batch_targets.to(device))
        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
        optimizer.step()

        
        accuracy = float(torch.sum(torch.argmax(predictions, dim=1) == batch_targets.to(device)).item()) / config.batch_size
        loss = loss.item()
        
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)
        
        if step % 10 == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
            ))
            save.append(accuracy)


        if (accuracy > highest):
            highest = accuracy
            torch.save(model.state_dict(), 'lstm-model.pt')

        if step == config.train_steps:
            break

    print('Done training.')
    return accuracy, highest, save

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_dim', type=int, default=100, help='Dimensionality of input sequence')
    parser.add_argument('--embed_dim', type=int, default=100, help='Dimensionality of the embeddings')
    parser.add_argument('--output_dim', type=int, default=1, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=256, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for Adam')
    parser.add_argument('--dropout', type=float, default=0.5, help='Drop out rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--bidirectional', type=bool, default=True)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    train(config)