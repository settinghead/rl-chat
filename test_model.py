import tensorflow as tf
#from policy_net import Encoder, Decoder
import numpy as np
#import tensorflow.contrib.eager as tfe
import functools
from encoder_decoder import Encoder, Decoder
import os
import time

def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)

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

if __name__ == "__main__":
    tf.enable_eager_execution()

    from utils import random_utterance
    samples = [random_utterance(4, 10) for _ in range(10000)]

    BATCH_SIZE = 64
    embedding_dim = 256
    units = 1024
  
    inp_lang = LanguageIndex(samples)
    targ_lang = LanguageIndex(samples)

    vocab_inp_size = len(inp_lang.word2idx)
    vocab_tar_size = len(targ_lang.word2idx)

    optimizer = tf.train.AdamOptimizer()
    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
    decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
    EPOCHS = 10

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                    encoder=encoder,
                                    decoder=decoder)

    input_tensor = [[inp_lang.word2idx[s] for s in sp.split(' ')] for sp in samples]
    target_tensor = [[targ_lang.word2idx[s] for s in sp.split(' ')] for sp in samples]
    
    # Calculate max_length of input and output tensor
    # Here, we'll set those to the longest sentence in the dataset
    max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)
    
    # Padding the input and output tensor to the maximum length
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, 
                                                                 maxlen=max_length_inp,
                                                                 padding='post')
    
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor, 
                                                                  maxlen=max_length_tar, 
                                                                  padding='post')
    BUFFER_SIZE = len(input_tensor)
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    N_BATCH = BUFFER_SIZE // BATCH_SIZE
    
    for epoch in range(EPOCHS):
        start = time.time()

        hidden = encoder.initialize_hidden_state()
        total_loss = 0
        
        for (batch, (inp, targ)) in enumerate(dataset):
            loss = 0
            
            with tf.GradientTape() as tape:
                enc_output, enc_hidden = encoder(inp, hidden)
                
                dec_hidden = enc_hidden
                
                dec_input = tf.expand_dims([targ_lang.word2idx['<start>']] * BATCH_SIZE, 1)       
                
                # Teacher forcing - feeding the target as the next input
                for t in range(1, targ.shape[1]):

                    predictions, dec_hidden = decoder(dec_input, dec_hidden)
                    loss += loss_function(targ[:, t], predictions)
                    
                    # using teacher forcing
                    dec_input = tf.expand_dims(targ[:, t], 1)
            
            batch_loss = (loss / int(targ.shape[1]))
            
            total_loss += batch_loss
            
            variables = encoder.variables + decoder.variables
            
            gradients = tape.gradient(loss, variables)
            
            optimizer.apply_gradients(zip(gradients, variables))
            
            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                            batch,
                                                            batch_loss.numpy()))
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
        
        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / N_BATCH))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
