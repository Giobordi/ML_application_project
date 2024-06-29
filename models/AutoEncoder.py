import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self, window_size):
        super(Encoder, self).__init__()
        # n_feature = 86
        # Input size: (n_feature, window_size)
        self.conv1d_1 = tf.keras.layers.Conv1D(filters=window_size * 2, kernel_size=7, strides=1, padding='valid')
        # size: (80, window_size*2)
        self.conv1d_2 = tf.keras.layers.Conv1D(filters=window_size * 4, kernel_size=4, strides=2, padding='same')
        # size: (40, window_size*4)
        self.conv1d_3 = tf.keras.layers.Conv1D(filters=window_size * 8, kernel_size=4, strides=2, padding='same')
        # size: (20, window_size*8)
        self.conv1d_4 = tf.keras.layers.Conv1D(filters=window_size * 16, kernel_size=4, strides=2, padding='same')
        # size: (10, window_size*16)
        self.conv1d_5 = tf.keras.layers.Conv1D(filters=window_size * 32, kernel_size=10, strides=1, padding='valid')
        # size: (1, window_size*32)

        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()
        self.batch_norm_3 = tf.keras.layers.BatchNormalization()
        self.batch_norm_4 = tf.keras.layers.BatchNormalization()

    def build(self, input_shape):
        super(Encoder, self).build(input_shape)

    def call(self, X):
        X = self.conv1d_1(X)
        X = self.batch_norm_1(X)
        X = tf.nn.leaky_relu(X)

        X = self.conv1d_2(X)
        X = self.batch_norm_2(X)
        X = tf.nn.leaky_relu(X)

        X = self.conv1d_3(X)
        X = self.batch_norm_3(X)
        X = tf.nn.leaky_relu(X)

        X = self.conv1d_4(X)
        X = self.batch_norm_4(X)
        X = tf.nn.leaky_relu(X)

        X = self.conv1d_5(X)

        return X


class Decoder(tf.keras.Model):
    def __init__(self, window_size):
        super(Decoder, self).__init__()
        # Input size: (1, window_size*32)
        self.conv_transpose_1d_1 = tf.keras.layers.Conv1DTranspose(filters=window_size * 16, kernel_size=10, strides=1,
                                                                   padding='valid')
        # size: (10, window_size*16)
        self.conv_transpose_1d_2 = tf.keras.layers.Conv1DTranspose(filters=window_size * 8, kernel_size=4, strides=2,
                                                                   padding='same')
        # size: (20, window_size*8)
        self.conv_transpose_1d_3 = tf.keras.layers.Conv1DTranspose(filters=window_size * 4, kernel_size=4, strides=2,
                                                                   padding='same')
        # size: (40, window_size*4)
        self.conv_transpose_1d_4 = tf.keras.layers.Conv1DTranspose(filters=window_size * 2, kernel_size=4, strides=2,
                                                                   padding='same')
        # size: (80, window_size*2)
        self.conv_transpose_1d_5 = tf.keras.layers.Conv1DTranspose(filters=window_size, kernel_size=7, strides=1,
                                                                   padding='valid')
        # size: (86, window_size)

        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()
        self.batch_norm_3 = tf.keras.layers.BatchNormalization()
        self.batch_norm_4 = tf.keras.layers.BatchNormalization()

    def build(self, input_shape):
        super(Decoder, self).build(input_shape)

    def call(self, X):
        X = self.conv_transpose_1d_1(X)
        X = self.batch_norm_1(X)
        X = tf.nn.relu(X)

        X = self.conv_transpose_1d_2(X)
        X = self.batch_norm_2(X)
        X = tf.nn.relu(X)

        X = self.conv_transpose_1d_3(X)
        X = self.batch_norm_3(X)
        X = tf.nn.relu(X)

        X = self.conv_transpose_1d_4(X)
        X = self.batch_norm_4(X)
        X = tf.nn.relu(X)

        X = self.conv_transpose_1d_5(X)

        return X


class AutoEncoder(tf.keras.Model):
    def __init__(self, window_size):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(window_size)
        self.decoder = Decoder(window_size)

    def build(self, input_shape):
        super(AutoEncoder, self).build(input_shape)

    def call(self, X):
        latent = self.encoder(X)
        X = self.decoder(latent)

        return X
