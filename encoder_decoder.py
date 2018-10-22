import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import utils

BEGIN_TAG = '<GO>'
END_TAG = '<EOS>'


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


def bilstm(units):
    if tf.test.is_gpu_available():
        return tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(units,
                                                                       return_sequences=True,
                                                                       return_state=True,
                                                                       dropout=0.25,
                                                                       recurrent_dropout=0.25))
    else:
        return tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units,
                                                                  return_sequences=True,
                                                                  return_state=True,
                                                                  dropout=0.25,
                                                                  recurrent_dropout=0.25,
                                                                  recurrent_activation='sigmoid'))


class GloVeEmbedding(tf.keras.Model):
    def __init__(
            self,
            vocab,
            embedding_dim=300):
        super(GloVeEmbedding, self).__init__()
        self.GloVe = tf.Variable(
            utils.get_GloVe_embeddings(vocab, embedding_dim), dtype='float32'
        )
        self.embedding_dim = embedding_dim

    def call(self, x):
        return tf.nn.embedding_lookup(self.GloVe, x)


class Encoder(tf.keras.Model):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        enc_units,
        batch_sz,
        use_GloVe=False,
        inp_lang=None,
        use_bilstm=False
    ):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        if use_GloVe:
            self.embedding = GloVeEmbedding(inp_lang, embedding_dim)
        else:
            self.embedding = tf.keras.layers.Embedding(
                vocab_size, embedding_dim)
        self.use_bilstm = use_bilstm
        self.gru = gru(self.enc_units)
        if use_bilstm:
            self.bilstm = bilstm(self.enc_units)

    def call(self, x, hidden):
        x = self.embedding(x)
        if self.use_bilstm:
            _, forward_h, forward_c, backward_h, backward_c = self.bilstm(
                x, initial_state=hidden)
            state_h = tf.keras.layers.Concatenate([forward_h, backward_h])
            state_c = tf.keras.layers.Concatenate([forward_c, backward_c])
            state = [state_h, state_c]
        else:
            _, state = self.gru(x, initial_state=hidden)
        return state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class Decoder(tf.keras.Model):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        dec_units,
        batch_sz,
        use_GloVe=False,
        targ_lang=None,
        use_bilstm=False
    ):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.vocab_size = vocab_size
        if use_GloVe:
            self.embedding = GloVeEmbedding(targ_lang, embedding_dim)
        else:
            self.embedding = tf.keras.layers.Embedding(
                vocab_size, embedding_dim)
        self.use_bilstm = use_bilstm
        self.gru = gru(self.dec_units)
        if use_bilstm:
            self.bilstm = bilstm(self.dec_units)
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x, hidden):
        x = self.embedding(x)
        if self.use_bilstm:
            output, forward_h, forward_c, backward_h, backward_c = self.bilstm(
                x, initial_state=hidden)
            state_h = tf.keras.layers.Concatenate([forward_h, backward_h])
            state_c = tf.keras.layers.Concatenate([forward_c, backward_c])
            state = [state_h, state_c]
        else:
            output, state = self.gru(x, initial_state=hidden)
        x = self.fc(output)
        x = tf.reshape(x, [x.shape[0], self.vocab_size])
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
        inp_lang,
        targ_lang,
        use_GloVe=False,
        display_result=False
    ):

        super(Seq2Seq, self).__init__()
        self.vocab_inp_size = vocab_inp_size
        self.vocab_tar_size = vocab_tar_size
        self.embedding_dim = embedding_dim
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.targ_lang = targ_lang
        self.encoder = Encoder(vocab_inp_size, embedding_dim,
                               enc_units, batch_sz, use_GloVe, inp_lang.vocab)
        self.decoder = Decoder(vocab_tar_size, embedding_dim,
                               enc_units, batch_sz, use_GloVe, targ_lang.vocab)
        self.hidden = self.encoder.initialize_hidden_state()
        self.display_result = display_result

    def loss_function(self, real, pred):
        mask = 1 - np.equal(real, 0)
        loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=real, logits=pred) * mask
        return tf.reduce_mean(loss_)

    def call(self, inp, targ):
        loss = 0
        enc_hidden = self.encoder(inp, self.hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims(
            [self.targ_lang.word2idx[BEGIN_TAG]] * self.batch_sz, 1)
        result = ''
        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            predictions, dec_hidden = self.decoder(dec_input, dec_hidden)
            loss += self.loss_function(targ[:, t], predictions)
            predicted_id = tf.argmax(predictions[0]).numpy()
            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)
            if self.targ_lang.idx2word[predicted_id] == END_TAG:
                if self.display_result:
                    print("result: ", result)
                return loss
            else:
                result += ' ' + self.targ_lang.idx2word[predicted_id]
        return loss

    """
    def run_epsiode(self, init_state):
        #Encode FULL state using instance of encoder
        #Then decode by manually looping
        #Simultanously generating samples from policy (Monte-Carlo)
        
        loss = 0
        enc_output, enc_hidden = self.encoder(init_state, self.hidden)
        dec_hidden = enc_hidden
        
        outputs = []
        curr_w = tf.expand_dims(
            [self.targ_lang.word2idx[BEGIN_TAG]] * self.batch_sz, 1)
            
        while curr_w != tf.expand_dims([self.targ_lang.word2idx[END_TAG]] * self.batch_sz, 1):  
            #EOS depends on granularity of responses (i.e. token, utterance, ...), like START. TODO: determine length of responses (episodes)
            
            w_probs, dec_hidden = decoder(curr_w, dec_hidden)
            w_probs = softmax(w_probs) # TODO: check if softmax is necessary
            dist = tf.distributions.Categorical(w_probs)
            curr_w = dist.sample()
            outputs.append(curr_w)
            
        return outputs[]
    """


def evaluate(model: Seq2Seq, eval_dataset):
    """evaluate an epoch."""
    total_loss = 0
    model.display_result = True
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
                print('batch {} training loss {:.4f}'.format(
                    batch, total_loss.numpy()))

    return total_loss
