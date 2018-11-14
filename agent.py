import tensorflow as tf


def Baseline():
    return tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='relu')
    ])
