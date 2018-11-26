import tensorflow as tf


def Baseline(num_hidden: int):
    return tf.keras.models.Sequential([
        tf.keras.layers.Dense(num_hidden, activation='relu'),
        tf.keras.layers.Dense(1, activation='relu')
    ])
