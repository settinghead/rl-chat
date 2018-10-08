import tensorflow as tf
from policy_net import Encoder, Decoder
import numpy as np
import tensorflow.contrib.eager as tfe
import functools


def loss(labels, predictions):
    """Computes mean squared loss."""
    return tf.reduce_mean(tf.square(predictions - labels))

    # some test code
if __name__ == "__main__":
    tf.enable_eager_execution()

    encoder = Encoder()
    decoder = Decoder()

    from utils import random_utterance
    samples = [random_utterance(4, 10) for _ in range(10000)]
    print(f"Obtained {len(samples)} samples.")
