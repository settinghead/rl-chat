import tensorflow as tf
import numpy as np
import time
import utils
from embedding_utils import get_GloVe_embeddings
from data import BEGIN_TAG, END_TAG
import beam_search


def gru(units):
    if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNGRU(units,
                                        return_sequences=True,
                                        return_state=True)
    else:
        return tf.keras.layers.GRU(units,
                                   return_sequences=True,
                                   return_state=True,
                                   dropout=0.3,
                                   recurrent_dropout=0.5)


def bilstm(units):
    if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNLSTM(units,
                                        return_sequences=True,
                                        return_state=True)
    else:
        return tf.keras.layers.LSTM(units,
                                    return_sequences=True,
                                    return_state=True,
                                    dropout=0.3,
                                    recurrent_dropout=0.3)


class GloVeEmbedding(tf.keras.Model):
    def __init__(
            self,
            vocab,
            embedding_dim=300,
            trainable=True):
        super(GloVeEmbedding, self).__init__()
        self.GloVe = tf.Variable(
            get_GloVe_embeddings(vocab, embedding_dim), dtype='float32',
            trainable=trainable
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
            self.embedding = GloVeEmbedding(
                inp_lang, embedding_dim, trainable=True)
        else:
            self.embedding = tf.keras.layers.Embedding(
                vocab_size, embedding_dim)
        self.use_bilstm = use_bilstm
        self.encode_model = gru(self.enc_units)
        if use_bilstm:
            self.encode_model = bilstm(self.enc_units)

    def call(self, x, hidden):
        x = self.embedding(x)
        if self.use_bilstm:
            _, state_h, state_c = self.encode_model(x, initial_state=[hidden, hidden])
            state = [state_h, state_c]
        else:
            _, state = self.encode_model(x, initial_state=hidden)
        return state


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
            self.embedding = GloVeEmbedding(
                targ_lang, embedding_dim, trainable=True)
        else:
            self.embedding = tf.keras.layers.Embedding(
                vocab_size, embedding_dim)

        self.use_bilstm = use_bilstm

        self.decode_model = gru(self.dec_units)
        if use_bilstm:
            self.decode_model = bilstm(self.dec_units)

        self.dropout = tf.keras.layers.Dropout(0.3)
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x, hidden):
        x = self.embedding(x)
        if self.use_bilstm:
            output, state_h, state_c = self.decode_model(x, initial_state=hidden)
            state = [state_h, state_c]
        else:
            output, state = self.decode_model(x, initial_state=hidden)
        output = self.dropout(output)
        x = self.fc(output)
        predicts = tf.nn.softmax(x)
        return predicts, state

TEACHER_FORCING = "TF"
BASIC = "B"
BEAM_SEARCH = "BS"

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
        max_length_tar,
        use_GloVe=False,
        mode=BEAM_SEARCH,
        use_bilstm = False,
        beam_size = 2
    ):

        super(Seq2Seq, self).__init__()
        self.vocab_inp_size = vocab_inp_size
        self.vocab_tar_size = vocab_tar_size
        self.embedding_dim = embedding_dim
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.targ_lang = targ_lang
        self.encoder = Encoder(vocab_inp_size, embedding_dim,
                               enc_units, batch_sz, use_GloVe, inp_lang.vocab, use_bilstm=use_bilstm)
        self.decoder = Decoder(vocab_tar_size, embedding_dim,
                               enc_units, batch_sz, use_GloVe, targ_lang.vocab, use_bilstm=use_bilstm)
        self.hidden = tf.zeros((batch_sz, enc_units))
        self.max_length_tar = max_length_tar
        self.mode = mode
        self.beam_size = beam_size
        self.use_bilstm = use_bilstm
        self.bs = beam_search.BeamSearch(self.beam_size,
                                        self.targ_lang.word2idx[BEGIN_TAG],
                                        self.targ_lang.word2idx[END_TAG],
                                        self.targ_lang,
                                        self.max_length_tar,
                                        self.batch_sz,
                                        self.decoder)

    def loss_function(self, real, pred):
        # if it's PAD, loss is 0
        mask = 1 - np.equal(real, 0)
        loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=real, logits=pred) * mask
        return tf.reduce_mean(loss_)

    def call(self, inp, targ):
        loss = 0
        enc_hidden = self.encoder(inp, self.hidden)
        dec_hidden = enc_hidden

        if self.mode == BEAM_SEARCH:

            dec_input = tf.expand_dims(
                    [self.targ_lang.word2idx[BEGIN_TAG]]*self.batch_sz, 1)
            dec_hidden_copy = dec_hidden
            for t in range(1, targ.shape[1]):
                predictions, dec_hidden = self.decoder(dec_input, dec_hidden)
                predictions = tf.squeeze(predictions, axis=1)
                labels = []
                for i in range(self.batch_sz):
                    new_input = tf.reshape(dec_input[i], (1, 1))
                    if self.use_bilstm:
                        new_dec_hidden = [
                            tf.reshape(dec_hidden_copy[0][i], (1, self.enc_units)),
                            tf.reshape(dec_hidden_copy[1][i], (1, self.enc_units))]
                        best_beam = self.bs.beam_search(new_input, new_dec_hidden, lstm=True)
                    else:
                        new_dec_hidden = tf.reshape(dec_hidden_copy[i], (1, self.enc_units))
                        best_beam = self.bs.beam_search(new_input, new_dec_hidden)
                        
                    label = best_beam.tokens[1]
                    labels.append(label)
                dec_input = tf.expand_dims(labels, 1)
                loss += self.loss_function(tf.convert_to_tensor(labels), predictions)
                
            return loss

        if self.mode == BASIC:

            dec_input = tf.expand_dims(
                    [self.targ_lang.word2idx[BEGIN_TAG]]*self.batch_sz, 1)
            for t in range(1, targ.shape[1]):
                predictions, dec_hidden = self.decoder(dec_input, dec_hidden)
                predictions = tf.squeeze(predictions, axis=1)
                dec_input = tf.reshape(tf.argmax(predictions, axis=1), (self.batch_sz, 1))
                loss += self.loss_function(targ[:, t], predictions)
            return loss

        elif self.mode == TEACHER_FORCING:

            dec_input = tf.expand_dims(
                    [self.targ_lang.word2idx[BEGIN_TAG]] * self.batch_sz, 1)
            for t in range(1, targ.shape[1]):
                # Teacher forcing - feeding the target as the next input
                predictions, dec_hidden = self.decoder(dec_input, dec_hidden)
                dec_input = tf.expand_dims(targ[:, t], 1)
                predictions = tf.squeeze(predictions, axis=1)
                loss += self.loss_function(targ[:, t], predictions)
            return loss

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
            enc_hidden = model.encoder(inp, model.hidden)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims(
                [model.targ_lang.word2idx[BEGIN_TAG]] * model.batch_sz , 1)
            result = ""
            for t in range(model.max_length_tar):
                predictions, dec_hidden = model.decoder(dec_input, dec_hidden)
                predictions = tf.squeeze(predictions, axis=1)
                predicted_id = np.argmax(predictions[0])
                result += model.targ_lang.idx2word[predicted_id] + ' '
                if model.targ_lang.idx2word[predicted_id] == END_TAG:
                    print("result: ", result.replace(END_TAG, ""))
                else:
                    print(result)
                dec_input = tf.expand_dims([predicted_id] * model.batch_sz, 1)
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
