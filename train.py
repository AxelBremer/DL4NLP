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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the device which to run the model on
    device = torch.device(device)

   
    # Initialize the dataset and data loader (note the +1)
    dataset = IMDBDataset(train_or_test='train', seq_length=config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size, shuffle=True, num_workers=1)

    # Initialize the dataset and data loader (note the +1)
    test_dataset = IMDBDataset(train_or_test='test', seq_length=config.seq_length)
    test_data_loader = DataLoader(dataset, config.batch_size, shuffle=True, num_workers=1)

     # Initialize the model that we are going to use
    model = NN(dataset.vocab_size, config.embed_dim, config.hidden_dim, config.output_dim, config.n_layers, config.bidirectional, config.dropout, 0).to(device)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())#, lr=config.learning_rate)
    lowest = 100
    save = []
    epochs = 0

    while epochs < config.train_epochs:
        accuracies = []
        losses = [] 
        t1 = time.time()
        print('Training')
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):

            x = batch_inputs.long().to(device)
            y_target = batch_targets.long().to(device)

            predictions = model(x)


            loss = criterion(predictions, y_target)
            optimizer.zero_grad()
            loss.backward()
            
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
            optimizer.step()

            # print(predictions)

            
            accuracy = (torch.argmax(predictions, dim=1) == y_target).cpu().numpy().mean()
            loss = loss.item()

            accuracies.append(accuracy)
            losses.append(loss)


        t2 = time.time()
        examples_per_second = len(dataset)/float(t2-t1)

        accuracy = np.array(accuracies).mean()
        loss = np.array(losses).mean()

        # Test on test set
        print('Testing')
        with torch.no_grad():
            test_accuracies = []
            test_losses = []
            for step, (batch_inputs, batch_targets) in enumerate(test_data_loader):

                x = batch_inputs.long().to(device)
                y_target = batch_targets.long().to(device)

                predictions = model(x, dropout=False)

                test_loss = criterion(predictions, y_target)
                
                test_accuracy = (torch.argmax(predictions, dim=1) == y_target).cpu().numpy().mean()
                test_loss = test_loss.item()

                test_accuracies.append(test_accuracy)
                test_losses.append(test_loss)

        test_accuracy = np.array(test_accuracies).mean()
        test_loss = np.array(test_losses).mean()

        if (test_loss < lowest):
            lowest = test_loss
            torch.save(model.state_dict(), f'{config.seq_length}_model.pt')

        epochs += 1
        print("[{}] Train epochs {:04d}/{:04d}, Train Accuracy = {:.2f}, Train Loss = {:.3f}, Test Accuracy = {:.2f}, Test Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), epochs,
                        config.train_epochs, accuracy, loss,
                        test_accuracy, test_loss))

    print('Done training.')
    return accuracy, lowest, save

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--seq_length', type=int, default=200, help='Dimensionality of input sequence')
    parser.add_argument('--embed_dim', type=int, default=300, help='Dimensionality of the embeddings')
    parser.add_argument('--output_dim', type=int, default=2, help='Dimensionality of output sequence')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Number of hidden units in the model')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=256, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.5, help='Learning rate for Adam')
    parser.add_argument('--dropout', type=float, default=0.2, help='Drop out rate')
    parser.add_argument('--train_epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--bidirectional', type=bool, default=False)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    train(config)