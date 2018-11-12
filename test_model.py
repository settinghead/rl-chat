import tensorflow as tf
import numpy as np
import os
import time
import data
from corpus_utils import LanguageIndex, tokenize_sentence
from utils import max_length, load_trained_model

tf.enable_eager_execution()

BATCH_SIZE = 64
embedding_dim = 256
units = 1024

questions, answers = data.load_conv_text()
inp_lang = LanguageIndex(questions)
targ_lang = LanguageIndex(answers)

input_tensor = [[inp_lang.word2idx[token]
                 for token in tokenize_sentence(question)] for question in questions]
target_tensor = [[targ_lang.word2idx[token]
                  for token in tokenize_sentence(answer)] for answer in answers]

max_length_inp, max_length_tar = max_length(
    input_tensor), max_length(target_tensor)

model = load_trained_model(BATCH_SIZE, embedding_dim,
                           units, tf.train.AdamOptimizer())


def generate_answer(sentence, model, inp_lang, targ_lang, max_length_inp, max_length_tar):
    inputs = [inp_lang.word2idx[i] for i in tokenize_sentence(sentence)]
    inputs = tf.keras.preprocessing.sequence.pad_sequences(
        [inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = model.encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word2idx['<go>']], 0)

    for t in range(max_length_tar):
        predictions, dec_hidden = model.decoder(dec_input, dec_hidden)
        predicted_id = tf.argmax(predictions[0]).numpy()

        result += ' ' + targ_lang.idx2word[predicted_id]

        if targ_lang.idx2word[predicted_id] == '<eos>':
            return result, sentence

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence


result, sentence = generate_answer(
    "Where do you live?", model, inp_lang, targ_lang, max_length_inp, max_length_tar)
print("sentence :" + sentence)
print("result : " + result.replace('<eos>', ''))
print("----------")
result, sentence = generate_answer(
    "Where is Pasadena?", model, inp_lang, targ_lang, max_length_inp, max_length_tar)
print("sentence :" + sentence)
print("result : " + result.replace('<eos>', ''))
