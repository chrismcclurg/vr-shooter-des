import tensorflow as tf

class MLPQ(tf.keras.Model):
    def __init__(self, n_actions, n_hidden):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(n_hidden, activation='relu')
        self.d2 = tf.keras.layers.Dense(n_hidden, activation='relu')
        self.out = tf.keras.layers.Dense(n_actions)

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return self.out(x)