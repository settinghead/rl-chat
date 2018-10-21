import tensorflow as tf
import numpy as np
import encoder_decoder as encoder_decoder
import os
import time
import utils

tf.enable_eager_execution()
class LanguageIndex():
    def __init__(self, samples):
        self.samples = samples
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.create_index()

    def create_index(self):
        for phrase in self.samples:
            self.vocab.update(phrase.split(' '))

        self.vocab = sorted(self.vocab)

        self.word2idx['<pad>'] = 0
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1

        for word, index in self.word2idx.items():
            self.idx2word[index] = word


def max_length(tensor):
    return max(len(t) for t in tensor)


optimizer = tf.train.AdamOptimizer()
EPOCHS = 10000
BATCH_SIZE = 64
embedding_dim = 256
units = 1024

questions, answers = utils.load_conv_text()

BATCH_SIZE = 64
embedding_dim = 256
units = 1024

inp_lang = LanguageIndex(questions)
targ_lang = LanguageIndex(answers)

input_tensor = [[inp_lang.word2idx[s]
                     for s in sp.split(' ')] for sp in questions]
target_tensor = [[targ_lang.word2idx[s]
                      for s in sp.split(' ')] for sp in answers]

max_length_inp, max_length_tar = max_length(
        input_tensor), max_length(target_tensor)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
vocab_inp_size = len(inp_lang.word2idx)
vocab_tar_size = len(targ_lang.word2idx)
model = encoder_decoder.Seq2Seq(
        vocab_inp_size, vocab_tar_size, embedding_dim, units, BATCH_SIZE, targ_lang)
checkpoint = tf.train.Checkpoint(optimizer=optimizer, seq2seq=model)

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


def generate_answer(sentence, model, inp_lang, targ_lang, max_length_inp, max_length_tar):
    inputs = [inp_lang.word2idx[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    
    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = model.encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word2idx['<GO>']], 0)

    for t in range(max_length_tar):
        predictions, dec_hidden = model.decoder(dec_input, dec_hidden)
        predicted_id = tf.argmax(predictions[0]).numpy()

        result += ' ' + targ_lang.idx2word[predicted_id]

        if targ_lang.idx2word[predicted_id] == '<EOS>':
            return result, sentence
        
        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence

result, sentence = generate_answer("Where do you live?", model, inp_lang, targ_lang, max_length_inp, max_length_tar)
print("sentence :" + sentence)
print("result : " + result.replace('<EOS>',''))
print("----------")
result, sentence = generate_answer("Where is Pasadena?", model, inp_lang, targ_lang, max_length_inp, max_length_tar)
print("sentence :" + sentence)
print("result : " + result.replace('<EOS>',''))
