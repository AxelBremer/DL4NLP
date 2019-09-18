
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

################################################################################

class NN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.nn = nn.LSTM(input_size=embedding_dim, 
                          hidden_size=hidden_dim, 
                          num_layers=n_layers,
                          bidirectional=bidirectional)
        
        if bidirectional:
            self.fc = nn.Linear(hidden_dim*2, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, text, dropout=True):
        embeddings = self.embedding(text).transpose(1, 0)
        
        if dropout:
            lstm_output, (hidden, cell) = self.nn(self.dropout(embeddings))
            last_output = lstm_output[-1,:,:].squeeze()

            output = self.fc(self.dropout(last_output))
        else:
            lstm_output, (hidden, cell) = self.nn(embeddings)
            last_output = lstm_output[-1,:,:].squeeze()

            output = self.fc(last_output)

        return self.sigmoid(output)
