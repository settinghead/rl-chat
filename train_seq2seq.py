import tensorflow as tf
import numpy as np
import functools
import encoder_decoder as encoder_decoder
import os
import time
import utils
from corpus_utils import LanguageIndex, tokenize_sentence, EMPTY_IDX
import data
from sklearn.model_selection import train_test_split
from embedding_utils import get_embedding_dim

if __name__ == "__main__":
    tf.enable_eager_execution()

    USE_GLOVE = True

    # why not both?
    questions1, answers1 = data.load_conv_text()
    #questions2, answers2 = data.load_opensubtitles_text()
    questions2, answers2 = [], []
    questions = list(questions1) + list(questions2)
    answers = list(answers1) + list(answers2)
    # questions, answers = data.load_conv_text()

    inp_lang = LanguageIndex(questions)
    targ_lang = LanguageIndex(answers)

    BATCH_SIZE = 512

    EMBEDDING_DIM = get_embedding_dim(USE_GLOVE)
    units = 512

    print("Vocab size: ", len(inp_lang.vocab), len(targ_lang.vocab))

    vocab_inp_size = len(inp_lang.word2idx)
    vocab_tar_size = len(targ_lang.word2idx)

    optimizer = tf.train.AdamOptimizer()
    EPOCHS = 1000000

    input_tensor = [[inp_lang.word2idx[token] for token in tokenize_sentence(
        question)] for question in questions]
    target_tensor = [[targ_lang.word2idx[token]
                      for token in tokenize_sentence(answer)] for answer in answers]
    # Calculate max_length of input and output tensor
    # Here, we'll set those to the longest sentence in the dataset
    max_length_inp, max_length_tar = utils.max_length(
        input_tensor), utils.max_length(target_tensor)

    # Padding the input and output tensor to the maximum length
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor,
                                                                 maxlen=max_length_inp,
                                                                 padding='post',
                                                                 value=EMPTY_IDX)

    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor,
                                                                  maxlen=max_length_tar,
                                                                  padding='post',
                                                                  value=EMPTY_IDX)

    # Creating training and validation sets using an 80-20 split
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(
        input_tensor, target_tensor, test_size=0.2)

    BUFFER_SIZE = len(input_tensor_train)
    dataset = tf.data.Dataset.from_tensor_slices(
        (input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    EVAL_BUFFER_SIZE = len(input_tensor_val)
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (input_tensor_val, target_tensor_val)).shuffle(EVAL_BUFFER_SIZE)
    val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)
    N_BATCH = BUFFER_SIZE // BATCH_SIZE

    # model: encoder_decoder.Seq2Seq = utils.load_trained_model(
    #     BATCH_SIZE, embedding_dim, units, tf.train.AdamOptimizer())

    model = encoder_decoder.Seq2Seq(
        vocab_inp_size, vocab_tar_size, EMBEDDING_DIM, units, BATCH_SIZE,
        inp_lang=inp_lang, targ_lang=targ_lang,
        use_GloVe=USE_GLOVE,
        display_result=True
    )

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, seq2seq=model)

    for epoch in range(EPOCHS):
        start = time.time()
        train_total_loss = encoder_decoder.train(model, optimizer, dataset)
        eval_total_loss = encoder_decoder.evaluate(model, val_dataset)

        # saving (checkpoint) the model every 100 epochs
        if (epoch + 1) % 200 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time taken for epoch {}: {} sec\n'.format(
            epoch, time.time() - start)
        )
