
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
        
        self.nn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, text):
        
        #text = [sent len, batch size]
        
        embeddings = self.embedding(text)
        # embedded = self.dropout(embeddings)
        
        lstm_output, (hidden, cell) = self.nn(embeddings)

        output = self.fc(lstm_output[:,-1,:])

        return self.sigmoid(output)
