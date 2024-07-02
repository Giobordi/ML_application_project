import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import tensorflow as tf
from debugpy.common.log import log_dir
from keras.src.callbacks import History
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

from models.AdversarialAutoEncoder import AdversarialAutoEncoder
from models.AutoEncoder import AutoEncoder

load_dotenv(".env")


## Anomaly with label 1 
## Normal with label 0

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
    normal_data_windowed = sliding_window(df_kuka_normal, window_size, window_step_size)
    test_data_slow = sliding_window(df_kuka_slow, window_size, window_step_size)

    train_data, test_data_normal = train_test_split(normal_data_windowed, test_size=0.3)

    test_data = np.concatenate([test_data_slow, test_data_normal])
    test_data_labels = np.concatenate([
        np.ones(test_data_slow.shape[0]), # Anomalies
        np.zeros(test_data_normal.shape[0]), # Normal
    ])

    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}, "
          f"formed by: {test_data_slow.shape} anomalies and {test_data_normal.shape} normal")

    shuffled_indices = np.random.permutation(len(test_data))
    test_data_shuffled = test_data[shuffled_indices]
    test_data_labels_shuffled = test_data_labels[shuffled_indices]
    test_data_slow_tensor = tf.convert_to_tensor(test_data_slow)

    # AutoEncoder
    autoencoder, train_history_ae = init_and_train_autoencoder(train_data, validation, window_size, lr, steps_per_epoch,
                                                               epochs)
    print(train_history_ae.history)

    normal_reconstructions = autoencoder.predict(train_data)
    plot_reconstruction_error(train_data, normal_reconstructions)

    normal_train_loss = tf.keras.losses.MeanSquaredError().call(train_data, normal_reconstructions)
    mean_mse_normal = np.mean(normal_train_loss.numpy(), axis=1)
    # plot_loss(mean_mse)

    threshold = np.mean(mean_mse_normal) + np.std(mean_mse_normal)
    print("Threshold: ", threshold)

    reconstructions_slow = autoencoder.predict(test_data_slow_tensor)
    test_loss_slow = tf.keras.losses.MeanSquaredError().call(test_data_slow_tensor, reconstructions_slow)
    mean_mse_test_slow = np.mean(test_loss_slow.numpy(), axis=1)
    # plot_loss(mean_mse_test)

    plot_train_and_test_losses(mean_mse_normal, mean_mse_test_slow, threshold, x_min=3e-4, x_max=1e-3)

    preds = predict(autoencoder, test_data_shuffled, threshold)
    print_stats(preds, test_data_labels_shuffled)
    
    plot_confusion_matrix(preds, test_data_labels_shuffled)
    plot_roc_curve(preds, test_data_labels_shuffled)

    # Adversarial AutoEncoder
    aae, train_history_aae = init_and_train_aae(train_data, validation, window_size, lr, steps_per_epoch, epochs)
    print(train_history_aae.history)

    aae_normal_reconstructions = aae.predict(train_data)
    plot_reconstruction_error(train_data, aae_normal_reconstructions)

    aae_normal_train_loss = tf.keras.losses.MeanSquaredError().call(train_data, aae_normal_reconstructions)
    aae_mean_mse = np.mean(aae_normal_train_loss.numpy(), axis=1)
    # plot_loss(aae_mean_mse)

    aae_threshold = np.mean(aae_mean_mse) + np.std(aae_mean_mse)
    aae_test_reconstructions_slow = aae.predict(test_data_slow_tensor)

    aae_test_loss_slow = tf.keras.losses.MeanSquaredError().call(test_data_slow_tensor, aae_test_reconstructions_slow)
    aae_mean_mse_test_slow = np.mean(aae_test_loss_slow.numpy(), axis=1)
    # plot_loss(aae_mean_mse_test)

    plot_train_and_test_losses(aae_mean_mse, aae_mean_mse_test_slow, aae_threshold, x_min=3e-4, x_max=65e-4)

    aae_preds = predict(aae, test_data_shuffled, aae_threshold)
    print_stats(aae_preds, test_data_labels_shuffled)
    
    plot_confusion_matrix(aae_preds, test_data_labels_shuffled)
    plot_roc_curve(aae_preds, test_data_labels_shuffled)


def init_and_train_autoencoder(train_data, validation, window_size, lr, steps_per_epoch, epochs) -> (AutoEncoder, History):
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
        validation_dataset = tf.data.Dataset.from_tensor_slices((val_normal, val_normal)).batch(256)

        training_history = autoencoder.fit(train_dataset, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                           callbacks=[callback, tensorboard_callback],
                                           validation_data=validation_dataset)
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_data)).repeat().batch(256)
        training_history = autoencoder.fit(train_dataset, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                           callbacks=[callback, tensorboard_callback])

    autoencoder.summary()

    return autoencoder, training_history


def init_and_train_aae(train_data, validation, window_size, lr, steps_per_epoch, epochs) \
        -> (AdversarialAutoEncoder, History):
    aae: AdversarialAutoEncoder = AdversarialAutoEncoder(window_size)
    aae.build(train_data.shape)

    autoencoder_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    aae.compile(discriminator_optimizer, autoencoder_optimizer)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    if validation:
        train_normal, val_normal = train_test_split(train_data, test_size=0.2)

        train_dataset = tf.data.Dataset.from_tensor_slices((train_normal, train_normal)).repeat().batch(256)
        validation_dataset = tf.data.Dataset.from_tensor_slices((val_normal, val_normal)).batch(256)
        training_history = aae.fit(train_dataset, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                   callbacks=[tensorboard_callback], validation_data=validation_dataset)
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_data)).repeat().batch(256)
        training_history = aae.fit(train_dataset, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                   callbacks=[tensorboard_callback])

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


def plot_train_and_test_losses(mean_mse_train, mean_mse_test, threshold, x_min, x_max):
    plt.hist(mean_mse_train[:, None], bins=40, color="b") # , range=(x_min, x_max))
    plt.hist(mean_mse_test[:, None], bins=40, color="r")
    plt.axvline(x=threshold, color="green", linestyle="--")
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
    loss = tf.keras.losses.MeanSquaredError().call(data, reconstructions)
    mean_mse_test = np.mean(loss.numpy(), axis=1)
    return tf.math.greater(mean_mse_test, threshold)


def print_stats(predictions, labels):
    print("Accuracy = {}".format(accuracy_score(labels, predictions)))
    print("Precision = {}".format(precision_score(labels, predictions)))
    print("Recall = {}".format(recall_score(labels, predictions)))
    print("F1 = {}".format(f1_score(labels, predictions)))


def plot_confusion_matrix(predictions, labels):
    # False Positive: Normal data classified as anomaly
    # False Negative: Anomaly data classified as normal
    cm = confusion_matrix(labels, predictions.numpy(), labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot()
    plt.show()


def plot_roc_curve(predictions, labels):
    fpr, tpr, thresholds = roc_curve(labels, predictions.numpy())
    # Calculate AUC
    auc = roc_auc_score(labels, predictions.numpy())

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
