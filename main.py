import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import tensorflow as tf
from debugpy.common.log import log_dir
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

from models.AdversarialAutoEncoder import AdversarialAutoEncoder
from models.AutoEncoder import AutoEncoder

load_dotenv(".env")


def main():
    # Load config
    window_size = int(os.getenv("WINDOW_SIZE"))
    window_step_size = int(os.getenv("WINDOW_STEP_SIZE"))
    lr = float(os.getenv("LEARNING_RATE"))
    steps_per_epoch = int(os.getenv("STEPS_PER_EPOCH"))
    epochs = int(os.getenv("EPOCHS"))
    validation = os.getenv("VALIDATION").strip() == "True"

    # Load dataset
    df_kuka_normal, df_kuka_slow = load_dataset()
    train_data = sliding_window(df_kuka_normal, window_size, window_step_size)

    # AutoEncoder
    autoencoder, train_history_ae = init_and_train_autoencoder(train_data, validation, window_size, lr, steps_per_epoch, epochs)
    print(train_history_ae)

    normal_reconstructions = autoencoder.predict(train_data)
    plot_reconstruction_error(train_data, normal_reconstructions)

    normal_train_loss = tf.keras.losses.mse(normal_reconstructions, train_data)
    mean_mse = np.mean(normal_train_loss, axis=1)
    plot_loss(mean_mse)

    threshold = np.mean(mean_mse) + np.std(mean_mse)
    print("Threshold: ", threshold)

    test_data = sliding_window(df_kuka_slow, window_size, window_step_size)
    reconstructions = autoencoder.predict(test_data)

    test_loss = tf.keras.losses.mse(reconstructions, test_data)
    mean_mse_test = np.mean(test_loss, axis=1)
    plot_loss(mean_mse_test)

    preds = predict(autoencoder, test_data, threshold)
    print_stats(np.bitwise_not(preds), np.ones_like(preds))

    # Adversarial AutoEncoder
    aae, train_history_aae = init_and_train_aae(train_data, validation, window_size, lr, steps_per_epoch, epochs)
    print(train_history_aae)

    aae_normal_reconstructions = aae.autoencoder.predict(train_data)
    plot_reconstruction_error(train_data, aae_normal_reconstructions)

    aae_normal_train_loss = tf.keras.losses.mse(aae_normal_reconstructions, train_data)
    aae_mean_mse = np.mean(aae_normal_train_loss, axis=1)
    plot_loss(aae_mean_mse)

    aae_threshold = np.mean(aae_mean_mse) + np.std(aae_mean_mse)

    aae_preds = predict(aae.autoencoder, test_data, aae_threshold)
    print_stats(np.bitwise_not(aae_preds), np.ones_like(preds).astype(bool))



def init_and_train_autoencoder(train_data, validation, window_size, lr, steps_per_epoch, epochs) -> (AutoEncoder, any):
    autoencoder: AutoEncoder = AutoEncoder(window_size)
    autoencoder.build(train_data.shape)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    autoencoder.compile(optimizer=optimizer, loss='mse')
    training_history: any

    if validation:
        train_normal, val_normal = train_test_split(train_data, test_size=0.2)

        train_dataset = tf.data.Dataset.from_tensor_slices((train_normal, train_normal)).repeat().batch(256)
        validation_dataset = tf.data.Dataset.from_tensor_slices((val_normal, val_normal)).repeat().batch(256)

        training_history = autoencoder.fit(train_dataset, steps_per_epoch=steps_per_epoch, epochs=epochs,
                        callbacks=[callback, tensorboard_callback],
                        validation_data=validation_dataset)
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_data)).repeat().batch(256)
        training_history = autoencoder.fit(train_dataset, steps_per_epoch=steps_per_epoch, epochs=epochs,
                        callbacks=[callback, tensorboard_callback])

    autoencoder.summary()

    return autoencoder, training_history


def init_and_train_aae(train_data, validation, window_size, lr, steps_per_epoch, epochs) -> (AdversarialAutoEncoder, any):
    aae: AdversarialAutoEncoder = AdversarialAutoEncoder(window_size)
    aae.build(train_data.shape)

    autoencoder_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    aae.compile(discriminator_optimizer, autoencoder_optimizer)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    if validation:
        train_normal, val_normal = train_test_split(train_data, test_size=0.2)

        train_dataset = tf.data.Dataset.from_tensor_slices((train_normal, train_normal)).repeat().batch(256)
        validation_dataset = tf.data.Dataset.from_tensor_slices((val_normal, val_normal)).repeat().batch(256)
        training_history = aae.fit(train_dataset, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=[tensorboard_callback], validation_data=validation_dataset)
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_data)).repeat().batch(256)
        training_history = aae.fit(train_dataset, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=[tensorboard_callback])

    aae.summary()

    return aae, training_history


def plot_reconstruction_error(train_data, reconstructions):
    sample_inside_window = 4  # from 0 to 10
    window_number = 38000  # from 0 to the number of windows created

    # Plot the reconstruction error for each feature
    plt.plot(train_data[window_number, :, sample_inside_window], 'b')
    plt.plot(reconstructions[window_number, :, sample_inside_window], 'r')
    plt.fill_between(np.arange(86), reconstructions[window_number, :, sample_inside_window],
                     train_data[window_number, :, sample_inside_window], color='lightcoral')
    plt.legend(labels=["Input", "Reconstruction", "Error"])
    plt.show()


def plot_loss(mean_mse):
    plt.hist(mean_mse[:, None], bins=20)
    plt.xlabel("Loss")
    plt.ylabel("No of examples")
    plt.show()


def load_dataset():
    root_dir_dataset = os.getenv("ROOTDIR_DATASET")

    # Load normal dataset
    kuka_column_names_path = os.path.join(root_dir_dataset, 'KukaColumnNames.npy')
    kuka_column_names = np.load(kuka_column_names_path)
    kuka_column_names.reshape((1, -1))

    kuka_normal_path = os.path.join(root_dir_dataset, 'KukaNormal.npy')
    kuka_normal = np.load(kuka_normal_path)
    kuka_normal = np.hstack((kuka_normal, np.zeros(kuka_normal.shape[0]).reshape(-1, 1)))
    kuka_normal = kuka_normal.astype('float32')
    kuka_normal = (kuka_normal - np.min(kuka_normal)) / (np.max(kuka_normal) - np.min(kuka_normal))

    kuka_normal = np.vstack((kuka_column_names, kuka_normal))
    df_kuka_normal = pd.DataFrame(kuka_normal)
    df_kuka_normal.columns = df_kuka_normal.iloc[0]
    df_kuka_normal = df_kuka_normal.iloc[1:]

    df_kuka_normal.drop("anomaly", axis=1, inplace=True)
    df_kuka_normal = df_kuka_normal.reset_index(drop=True)

    # Load anomalies
    kuka_slow_path = os.path.join(root_dir_dataset, 'KukaSlow.npy')
    kuka_slow = np.load(kuka_slow_path)
    kuka_slow = kuka_slow.astype('float32')
    kuka_slow = (kuka_slow - np.min(kuka_slow)) / (np.max(kuka_slow) - np.min(kuka_slow))

    kuka_slow = np.vstack((kuka_column_names, kuka_slow))

    df_kuka_slow = pd.DataFrame(kuka_slow)
    df_kuka_slow.columns = df_kuka_slow.iloc[0]
    df_kuka_slow = df_kuka_slow.iloc[1:]
    df_kuka_slow = df_kuka_slow.reset_index(drop=True)
    df_kuka_slow.drop("anomaly", axis=1, inplace=True)

    return df_kuka_normal, df_kuka_slow


def sliding_window(data, window_size, step_size):
    windows = []
    for i in range(0, data.shape[0] - window_size + 1, step_size):
        windows.append(data.iloc[i:i + window_size].to_numpy().astype('float32'))

    x = np.array(windows).astype('float32')
    return np.swapaxes(x, 1, 2)


def predict(model, data, threshold):
    reconstructions = model(data)
    loss = tf.keras.losses.mse(reconstructions, data)
    mean_mse_test = np.mean(loss, axis=1)
    return tf.math.less(mean_mse_test, threshold)


def print_stats(predictions, labels):
    print("Accuracy = {}".format(accuracy_score(labels, predictions)))
    print("Precision = {}".format(precision_score(labels, predictions)))
    print("Recall = {}".format(recall_score(labels, predictions)))
    print("F1 = {}".format(f1_score(labels, predictions)))


if __name__ == '__main__':
    main()
