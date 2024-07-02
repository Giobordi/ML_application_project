import tensorflow as tf
from models.AutoEncoder import AutoEncoder


class Discriminator(tf.keras.Model):
    def __init__(self, window_size):
        super(Discriminator, self).__init__()
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

        self.flatten = tf.keras.layers.Flatten()

        self.dense_1 = tf.keras.layers.Dense(window_size * 8, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(window_size * 4, activation='relu')
        self.dense_3 = tf.keras.layers.Dense(window_size, activation='relu')

        self.softmax_1 = tf.keras.layers.Dense(1, activation='sigmoid')

        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()
        self.batch_norm_3 = tf.keras.layers.BatchNormalization()
        self.batch_norm_4 = tf.keras.layers.BatchNormalization()
        self.batch_norm_5 = tf.keras.layers.BatchNormalization()
        self.batch_norm_6 = tf.keras.layers.BatchNormalization()
        self.batch_norm_7 = tf.keras.layers.BatchNormalization()

    def build(self, input_shape):
        super(Discriminator, self).build(input_shape)

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
        X = self.batch_norm_5(X)
        X = tf.nn.leaky_relu(X)

        X = self.flatten(X)

        X = self.dense_1(X)
        X = self.batch_norm_6(X)

        X = self.dense_2(X)
        X = self.batch_norm_7(X)

        X = self.dense_3(X)
        X = self.softmax_1(X)

        return X


class AdversarialAutoEncoder(tf.keras.Model):
    def __init__(self, window_size):
        super(AdversarialAutoEncoder, self).__init__()
        self.autoencoder = AutoEncoder(window_size)
        self.discriminator = Discriminator(window_size)
        self.d_optimizer = None
        self.ae_optimizer = None

    def build(self, input_shape):
        super(AdversarialAutoEncoder, self).build(input_shape)

    def compile(self, d_optimizer, ae_optimizer):
        super(AdversarialAutoEncoder, self).compile()
        self.d_optimizer = d_optimizer
        self.ae_optimizer = ae_optimizer

    def call(self, data):
        return self.autoencoder.call(data)

    def train_step(self, data):
        real_data, _ = data

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_data = self.autoencoder.call(real_data)
            real_output = self.discriminator.call(real_data)
            fake_output = self.discriminator.call(generated_data)

            mse = tf.keras.losses.MeanSquaredError().call(real_data, generated_data)
            autoencoder_loss = tf.reduce_mean(tf.nn.relu(1 - fake_output)) + tf.reduce_mean(mse)

            discriminator_loss = tf.reduce_mean(tf.nn.relu(1 - real_output) + tf.nn.relu(1 + fake_output))

        gradients_of_generator = gen_tape.gradient(autoencoder_loss, self.autoencoder.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(discriminator_loss, self.discriminator.trainable_variables)

        self.ae_optimizer.apply_gradients(zip(gradients_of_generator, self.autoencoder.trainable_variables))
        self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return {
            "d_loss": discriminator_loss,
            "ae_loss": autoencoder_loss
        }
