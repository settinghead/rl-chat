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
        # TODO: incomplete
        self.encoder = Sequential([
            LSTM(64, return_state=True),
        ])

    def call(self, x, hidden):
        # TODO: incomplete
        output, state_h, state_c = self.encoder(x)
        encoder_states = [state_h, state_c]
        return output, encoder_states


class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        # TODO: incomplete
        self.decoder = Sequential([
            LSTM(64, return_sequences=True, return_state=True)
        ])

    def call(self, x, enc_state):
        x = self.decoder(x, initial_state=enc_state)
        # TODO: incomplete

# TODO: maybe a simple test case would help

# if __name__ == "__main__":
#     encoder = Encoder()
#     decoder = Decoder()
#     encoder_hidden = initial_hidden

#     for w_in in input_words:
#     _, encoder_hidden = encoder(w_in, encoder_hidden)

#     decoder_hidden = encoder_hidden
#     w = input_words[-1]
#     while w != EOS:
#     w, decoder_hidden = decoder(decoder_hidden, w)
#     words_so_far.append(w)
