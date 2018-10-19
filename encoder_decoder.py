import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time

def gru(units):
        if tf.test.is_gpu_available():
            return tf.keras.layers.CuDNNGRU(units, 
                                            return_sequences=True, 
                                            return_state=True, 
                                            recurrent_initializer='glorot_uniform')
        else:
            return tf.keras.layers.GRU(units, 
                                    return_sequences=True, 
                                    return_state=True, 
                                    recurrent_activation='sigmoid', 
                                    recurrent_initializer='glorot_uniform')

class Encoder(tf.keras.Model):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        enc_units,
        batch_sz
    ):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.enc_units)
        
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)        
        return output, state
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

class Decoder(tf.keras.Model):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        dec_units,
        batch_sz
    ):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.vocab_size = vocab_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.dec_units)
        self.fc = tf.keras.layers.Dense(vocab_size)
        
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        x = self.fc(output)
        x = tf.reshape(x, [self.batch_sz, self.vocab_size])
        logits = tf.nn.softmax(x)
        return logits, state
        
class Seq2Seq(tf.keras.Model):
    def __init__(
        self,
        vocab_inp_size,
        vocab_tar_size,
        embedding_dim,
        enc_units,
        batch_sz,
        targ_lang
    ):

        super(Seq2Seq, self).__init__()
        self.vocab_inp_size = vocab_inp_size
        self.vocab_tar_size = vocab_tar_size
        self.embedding_dim = embedding_dim
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.targ_lang = targ_lang
        self.encoder = Encoder(vocab_inp_size, embedding_dim, enc_units, batch_sz)
        self.decoder = Decoder(vocab_tar_size, embedding_dim, enc_units, batch_sz)
        self.hidden = self.encoder.initialize_hidden_state()

    def loss_function(self, real, pred):
        mask = 1 - np.equal(real, 0)
        loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
        return tf.reduce_mean(loss_)

    def call(self, inp, targ):
        loss = 0
        enc_output, enc_hidden = self.encoder(inp, self.hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.targ_lang.word2idx['<start>']] * self.batch_sz, 1)       
        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            predictions, dec_hidden = self.decoder(dec_input, dec_hidden)
            loss += self.loss_function(targ[:, t], predictions)
            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

        return loss


def evaluate(model: Seq2Seq, eval_dataset):
    """evaluate an epoch."""
    total_loss = 0
    for (batch, (inp, targ)) in enumerate(eval_dataset):
        loss = model(inp, targ)
        batch_loss = (loss / int(targ.shape[1]))
        total_loss += batch_loss
        if batch % 100 == 0:
            print('batch {} eval loss {:.4f}'.format(batch, total_loss.numpy()))
    
    return total_loss

def train(model: Seq2Seq, optimizer, train_dataset):
    """training an epoch."""
    total_loss = 0
    for (batch, (inp, targ)) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            loss = model(inp, targ)
            batch_loss = (loss / int(targ.shape[1]))
            total_loss += batch_loss
            variables = model.encoder.variables + model.decoder.variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))
            if batch % 100 == 0:
                print('batch {} training loss {:.4f}'.format(batch, total_loss.numpy()))

    return total_loss
