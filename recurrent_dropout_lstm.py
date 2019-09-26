import sys
sys.path.insert(0, 'recurrent_dropout_utils/')
from embed_dropout import embedded_dropout
from locked_dropout import LockedDropout
from weightdrop import WeightDrop
import numpy as np

import torch
import torch.nn as nn



class Model(nn.Module):
    def __init__(self, nb_words,embedding_size=300,  hidden_size=512, output_dim=2, n_layers=2,
                 wdrop=0.25, edrop=0.1, idrop=0.25, batch_first=True, bidirectional=False):
        super(Model, self).__init__()

        self.lockdrop = LockedDropout()
        self.bidirectional = bidirectional
        self.idrop = idrop
        self.edrop = edrop
        self.n_layers = n_layers
        self.embedding = nn.Embedding(nb_words, embedding_size,  padding_idx = 0)
        self.rnns = [
            nn.LSTM(embedding_size if l == 0 else hidden_size,
                   hidden_size, num_layers=1, batch_first=batch_first, bidirectional=False)
            for l in range(n_layers)
        ]
        if wdrop:
            self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop)
                         for rnn in self.rnns]
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.output_layer = nn.Linear(hidden_size, output_dim)
        if bidirectional: self.output_layer = nn.Linear(hidden_size*2, output_dim)
        self.init_weights()
        self.sigmoid = nn.Sigmoid()


    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.fill_(0)
        self.output_layer.weight.data.uniform_(-initrange, initrange)

    def lstm_pass(self, X):
        emb = embedded_dropout(self.embedding, X, dropout=self.edrop if self.training else 0)
        raw_output = self.lockdrop(emb, self.idrop)
        new_hidden, new_cell_state = [], []
        i = 0
        for l, rnn in enumerate(self.rnns):
            i+=1
            raw_output, (new_h, new_c) = rnn(raw_output)
            raw_output = self.lockdrop(raw_output, self.idrop) 
            new_hidden.append(new_h)
            new_cell_state.append(new_c) 

        hidden = torch.cat(new_hidden, 0)
        cell_state = torch.cat(new_cell_state, 0)
        final_output =  hidden[-1,:,:].squeeze()
        return final_output
        
    def forward(self, X):
        if not self.bidirectional:
            final_output = self.lstm_pass(X)
            final_output = self.output_layer(final_output)
        else:
            final_output_1 = self.lstm_pass(X)
            X2 = X.flip(1)
            final_output_2 = self.lstm_pass(X2)
            final_output = self.output_layer(torch.cat((final_output_1, final_output_2),1))
        
        return self.sigmoid(final_output).squeeze()


