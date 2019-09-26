
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import functional as F
################################################################################

class NN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        
        super().__init__()
        self.hidden_dim = hidden_dim 
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
        #self.attn = nn.Linear(hidden_dim + output_dim, 1)


        self.sigmoid = nn.Sigmoid()

    def attention_net(self, lstm_output, final_state):

        """ 
        Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
                  new hidden state.
                  
        Tensor Size :
                    hidden.size() = (batch_size, hidden_size)
                    attn_weights.size() = (batch_size, num_seq)
                    soft_attn_weights.size() = (batch_size, num_seq)
                    new_hidden_state.size() = (batch_size, hidden_size)
                      
        """
        hidden = final_state[-1]
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        
        return new_hidden_state


    def forward(self, text, dropout=True):
        embeddings = self.embedding(text).transpose(1, 0) 

        if dropout:
            lstm_output, (hidden, cell) = self.nn(self.dropout(embeddings))
            output = lstm_output.permute(1, 0, 2)
            last_output = lstm_output[-1,:,:].squeeze()
            
            attn_output = self.attention_net(output, hidden)
            output = self.fc(self.dropout(attn_output))
        else:
            lstm_output, (hidden, cell) = self.nn(embeddings)
            last_output = lstm_output[-1,:,:].squeeze()
            output = lstm_output.permute(1, 0, 2)

            attn_output = self.attention_net(output, hidden)
            output = self.fc(attn_output)


        return self.sigmoid(output)

    '''    
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
    '''