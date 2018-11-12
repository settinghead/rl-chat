import random
import string
import tensorflow as tf
import numpy as np
import encoder_decoder as encoder_decoder
import data
from corpus_utils import LanguageIndex, tokenize_sentence

# def get_ELMo_embeddings():
#     url = "https://tfhub.dev/google/elmo/2"
#     elmo = hub.Module(url)
#     return elmo


def max_length(tensor):
    return max(len(t) for t in tensor)


def load_trained_model(batch_size, embedding_dim, units, optimizer):
    questions, answers = data.load_conv_text()

    inp_lang = LanguageIndex(questions)
    targ_lang = LanguageIndex(answers)

    checkpoint_dir = './training_checkpoints'
    vocab_inp_size = len(inp_lang.word2idx)
    vocab_tar_size = len(targ_lang.word2idx)
    model = encoder_decoder.Seq2Seq(
        vocab_inp_size, vocab_tar_size, embedding_dim, units, batch_size, inp_lang, targ_lang)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, seq2seq=model)
    # restoring the latest checkpoint in checkpoint_dir
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    return model
