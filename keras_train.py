import numpy as np

from keras.preprocessing import sequence
from keras.preprocessing import text
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding, DropoutEmbedding
from keras.layers.recurrent import LSTM, GRU, DropoutLSTM, NaiveDropoutLSTM
from keras.callbacks import ModelCheckpoint, ModelTest
from keras.regularizers import l2

import torch
from torch import nn
from torch.utils.data import DataLoader
from imdb_dataset import IMDBDataset

seed = 0

print('Build model...')
model = Sequential()
model.add(DropoutEmbedding(nb_words + index_from, 128, W_regularizer=l2(weight_decay), p=p_emb))
model.add(DropoutLSTM(128, 128, truncate_gradient=maxlen, W_regularizer=l2(weight_decay), 
                      U_regularizer=l2(weight_decay), 
                      b_regularizer=l2(weight_decay), 
                      p_W=p_W, p_U=p_U))
model.add(Dropout(p_dense))
model.add(Dense(128, 1, W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay)))

#optimiser = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=False)
optimiser = 'adam'
model.compile(loss='mean_squared_error', optimizer=optimiser)


print("Train...")
checkpointer = ModelCheckpoint(filepath=folder+filename+".hdf5", 
    verbose=1, append_epoch_name=True, save_every_X_epochs=50)
modeltest_1 = ModelTest(X_train[:100], mean_y_train + std_y_train * np.atleast_2d(Y_train[:100]).T, 
                      test_every_X_epochs=1, verbose=0, loss='euclidean',
                      mean_y_train=mean_y_train, std_y_train=std_y_train, tau=0.1)
modeltest_2 = ModelTest(X_test, np.atleast_2d(Y_test).T, test_every_X_epochs=1, verbose=0, loss='euclidean',
                      mean_y_train=mean_y_train, std_y_train=std_y_train, tau=0.1)
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=250, 
          callbacks=[checkpointer, modeltest_1, modeltest_2]) #
# score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, show_accuracy=True)
# print('Test score:', score)
# print('Test accuracy:', acc)

# model.save_weights(folder+filename+"_250.hdf5", overwrite=True)



standard_prob = model.predict(X_train, batch_size=500, verbose=1)
print(np.mean(((mean_y_train + std_y_train * np.atleast_2d(Y_train).T) 
               - (mean_y_train + std_y_train * standard_prob))**2, 0)**0.5)


# In[ ]:

standard_prob = model.predict(X_test, batch_size=500, verbose=1)
#print(standard_prob)
T = 50
prob = np.array([model.predict_stochastic(X_test, batch_size=500, verbose=0) 
                 for _ in xrange(T)])
prob_mean = np.mean(prob, 0)
print(np.mean((np.atleast_2d(Y_test).T - (mean_y_train + std_y_train * standard_prob))**2, 0)**0.5)
print(np.mean((np.atleast_2d(Y_test).T - (mean_y_train + std_y_train * prob_mean))**2, 0)**0.5)