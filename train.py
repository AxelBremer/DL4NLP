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

from imdb_dataset import IMDBDataset
from model import NN

def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

   
    # Initialize the dataset and data loader (note the +1)
    dataset = IMDBDataset(train_or_test='train', seq_length=config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size, shuffle=True, num_workers=1)

     # Initialize the model that we are going to use
    model = NN(dataset.vocab_size, config.embed_dim, config.hidden_dim, config.output_dim, config.n_layers, config.bidirectional, config.dropout, 0).to(device)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.RMSprop(model.parameters()) # lr=config.learning_rate)
    highest = 0 
    save = []
    epochs = 0

    print('Starting training...')

    while epochs < config.train_epochs:

        t1 = time.time()

        for step, (batch_inputs, batch_targets) in enumerate(data_loader):

            x = batch_inputs.to(device)
            y_target = batch_targets.to(device)

            predictions = model(batch_inputs).to(device)


            loss = criterion(predictions, y_target)
            optimizer.zero_grad()
            loss.backward()
            
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
            optimizer.step()

            
            accuracy = (torch.argmax(predictions, dim=1) == batch_targets.to(device)).numpy().mean()
            loss = loss.item()
            
            if step % 100== 0:
                save.append(accuracy)


            if (accuracy > highest):
                highest = accuracy
                torch.save(model.state_dict(), 'lstm-model.pt')


        t2 = time.time()
        examples_per_second = len(dataset)/float(t2-t1)
        epochs += 1
        print("[{}] Train epochs {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), epochs,
                        config.train_epochs, config.batch_size, examples_per_second,
                        accuracy, loss))

    print('Done training.')
    return accuracy, highest, save

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--seq_length', type=int, default=30, help='Dimensionality of input sequence')
    parser.add_argument('--embed_dim', type=int, default=100, help='Dimensionality of the embeddings')
    parser.add_argument('--output_dim', type=int, default=2, help='Dimensionality of output sequence')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Number of hidden units in the model')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for Adam')
    parser.add_argument('--dropout', type=float, default=0.5, help='Drop out rate')
    parser.add_argument('--train_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--bidirectional', type=bool, default=False)
    parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    train(config)