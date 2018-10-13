import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.layers import (
    LSTM, Dense, Input
)
from keras import Model, Sequential


class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = Sequential([
            LSTM(64, return_state=True),
        ])

    def call(self, x, hidden):
        output, state_h, state_c = self.encoder(x)
        encoder_states = [state_h, state_c]
        return output, encoder_states


class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = Sequential([
            LSTM(64, return_sequences=True, return_state=True)
        ])

    def call(self, x, enc_state):
        x = self.decoder(x, initial_state=enc_state)
