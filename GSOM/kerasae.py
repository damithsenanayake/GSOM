from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
class AutoEncoder(object):

    def __init__(self, inp_size = 784, hid_size = 196):
        self.inp_size = inp_size
        self.hid_size = hid_size

        self.input_layer = Input(shape = (inp_size,))
        self.encoded  = Dense(hid_size, activation='relu')(self.input_layer)
        self.decoded = Dense(inp_size, activation='sigmoid')(self.encoded)

        self.model = Model(input = self.input_layer, output=self.decoded)

        self.encoder = Model(input = self.input_layer, output = self.encoded)
        encoded_input = Input(shape=(self.hid_size,))
        decoder_layer = self.model.layers[-1]
        self.decoder = Model(input = encoded_input, output = decoder_layer(encoded_input))
        self.model.compile(optimizer='adadelta', loss='binary_crossentropy')


    def fit_transform(self, X, epochs=5, lr = 0.1):
        if len(X.shape) ==1:
            X = np.matrix(X)
        self.model.optimizer.lr.set_value(lr)
        self.model.fit(X, X, nb_epoch=epochs, verbose=0)

        return self.decoder.predict(self.encoder.predict(X))

    def transform(self, X):
        # if len(X.shape) > 0:
        #     X = X[0]
        return self.decoder.predict(self.encoder.predict(X))

    def reduce(self, X):
        return self.encoder.predict(X)

    def w(self):
        return self.model.get_weights()

    def set_w(self, w):
        self.model.set_weights(w)