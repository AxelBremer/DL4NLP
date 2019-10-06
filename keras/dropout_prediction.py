'''
This file implements prediction with a Keras model using dropout. 
Normally dropout is automatically turned off when predicting.
'''

import numpy as np
import keras.backend as K

class KerasDropoutPrediction(object):
    def __init__(self,model):
        self.f = K.function(
                [model.layers[0].input, 
                 K.learning_phase()],
                [model.layers[-1].output])
                
    def predict(self,x, n_iter=10):
        return np.array(self.f([x,1]))

