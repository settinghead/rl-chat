import tensorflow as tf
import numpy as np
import encoder_decoder as encoder_decoder
import os
import time
import utils

tf.enable_eager_execution()

optimizer = tf.train.AdamOptimizer()
EPOCHS = 10000
BATCH_SIZE = 64
embedding_dim = 256
units = 1024

questions, answers = utils.load_conv_text()

BATCH_SIZE = 64
embedding_dim = 256
units = 1024

inp_lang = utils.LanguageIndex(questions)
targ_lang = utils.LanguageIndex(answers)

input_tensor = [[inp_lang.word2idx[token] for token in utils.tokenize_sentence(question)] for question in questions]
target_tensor = [[targ_lang.word2idx[token] for token in utils.tokenize_sentence(answer)] for answer in answers]

max_length_inp, max_length_tar = utils.max_length(
        input_tensor), utils.max_length(target_tensor)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
vocab_inp_size = len(inp_lang.word2idx)
vocab_tar_size = len(targ_lang.word2idx)
model = encoder_decoder.Seq2Seq(
        vocab_inp_size, vocab_tar_size, embedding_dim, units, BATCH_SIZE, inp_lang, targ_lang)
checkpoint = tf.train.Checkpoint(optimizer=optimizer, seq2seq=model)

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


def generate_answer(sentence, model, inp_lang, targ_lang, max_length_inp, max_length_tar):
    inputs = [inp_lang.word2idx[i] for i in utils.tokenize_sentence(sentence)]
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
