import os

import numpy as np
import tensorflow as tf
from debugpy.common.log import log_dir
from keras.src.callbacks import History
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from sklearn.model_selection import train_test_split

from models.AdversarialAutoEncoder import AdversarialAutoEncoder
from models.AutoEncoder import AutoEncoder
from utils.DataUtils import DataUtils
from utils.PlotUtils import PlotUtils


class ExperimentUtils:

    @staticmethod
    def ae_experiment(df_kuka_normal, df_kuka_slow, window_size, window_step_size, lr, steps_per_epoch, epochs,
                      validation):
        print(f"-----------------------------------------------------------------------------------")
        print(f"Running AE experiment with config: WS = {window_size}, lr = {lr}, epochs = {epochs}")
        print(f"-----------------------------------------------------------------------------------")

        architecture = "ae"
        data = DataUtils.prepare_experiment_data(df_kuka_normal, df_kuka_slow, window_size, window_step_size)
        train_data, test_data_shuffled_tensor, test_data_labels_shuffled, test_data_slow_tensor = data

        # AutoEncoder
        autoencoder, train_history_ae = _init_and_train_model(
            architecture, train_data, validation, window_size, lr, steps_per_epoch, epochs
        )
        print(train_history_ae.history)

        config = {
            "ARCHITECTURE": architecture,
            "EPOCHS": epochs,
            "LR": lr,
            "WINDOW_SIZE": window_size
        }

        return ExperimentUtils.run_experiment(autoencoder, train_data, test_data_slow_tensor, test_data_shuffled_tensor,
                                              test_data_labels_shuffled, config)

    @staticmethod
    def aae_experiment(df_kuka_normal, df_kuka_slow, window_size, window_step_size, lr, steps_per_epoch, epochs,
                       validation):
        print(f"-----------------------------------------------------------------------------------")
        print(f"Running AAE experiment with config: WS = {window_size}, lr = {lr}, epochs = {epochs}")
        print(f"-----------------------------------------------------------------------------------")

        architecture = "aae"
        data = DataUtils.prepare_experiment_data(df_kuka_normal, df_kuka_slow, window_size, window_step_size)
        train_data, test_data_shuffled_tensor, test_data_labels_shuffled, test_data_slow_tensor = data

        # Adversarial AutoEncoder
        aae, train_history_aae = _init_and_train_model(
            architecture, train_data, validation, window_size, lr, steps_per_epoch, epochs
        )
        print(train_history_aae.history)

        config = {
            "ARCHITECTURE": architecture,
            "EPOCHS": epochs,
            "LR": lr,
            "WINDOW_SIZE": window_size
        }

        return ExperimentUtils.run_experiment(aae, train_data, test_data_slow_tensor, test_data_shuffled_tensor,
                                              test_data_labels_shuffled, config)

    @staticmethod
    def run_experiment(model, train_data, test_data_slow_tensor, test_data_shuffled_tensor, test_data_labels_shuffled,
                       config):
        normal_reconstructions = model.predict(train_data)

        PlotUtils.plot_averaged_reconstruction_error(train_data, normal_reconstructions, config)

        normal_train_loss = tf.keras.losses.MeanSquaredError().call(train_data, normal_reconstructions)
        mean_mse = np.mean(normal_train_loss.numpy(), axis=1)
        # plot_loss(mean_mse)

        threshold = np.mean(mean_mse) + 3 * np.std(mean_mse)
        test_reconstructions_slow = model.predict(test_data_slow_tensor)

        test_loss_slow = tf.keras.losses.MeanSquaredError().call(test_data_slow_tensor, test_reconstructions_slow)
        mean_mse_test_slow = np.mean(test_loss_slow.numpy(), axis=1)
        # plot_loss(mean_mse_test)

        PlotUtils.plot_train_and_test_losses(mean_mse, mean_mse_test_slow, threshold, config)

        preds = _predict(model, test_data_shuffled_tensor, threshold)
        f1_5 = _print_stats(preds, test_data_labels_shuffled, config)

        PlotUtils.plot_confusion_matrix(preds, test_data_labels_shuffled, config)
        PlotUtils.plot_roc_curve(preds, test_data_labels_shuffled, config)

        return f1_5


def _predict(model, data, threshold):
    reconstructions = model(data)
    loss = tf.keras.losses.MeanSquaredError().call(data, reconstructions)
    mean_mse_test = np.mean(loss.numpy(), axis=1)
    return tf.math.greater(mean_mse_test, threshold)


def _print_stats(predictions, labels, config):
    accuracy = accuracy_score(labels, predictions)
    prediction = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1_sc = f1_score(labels, predictions)
    print(f"Accuracy = {accuracy}")
    print(f"Precision = {prediction}")
    print(f"Recall = {recall}")
    print(f"F1 = {f1_sc}")
    # beta = 1.5 makes recall one time and a half as important as precision
    f1_5 = fbeta_score(labels, predictions, beta=1.5)
    print(f"F1.5 = {f1_5}")

    if not os.path.exists(f"{config['ARCHITECTURE']}/results"):
        os.makedirs(f"{config['ARCHITECTURE']}/results")

    with open(f"{config['ARCHITECTURE']}/results/logging_file.txt", "a+") as f:
        f.write(f"Windows_size = {config['WINDOW_SIZE']}  LR = {config['LR']}  Epochs = {config['EPOCHS']}\n"
                f"\tAccuracy = {accuracy}\n"
                f"\tPrecision = {prediction}\n"
                f"\tRecall = {recall}\n"
                f"\tF1 = {f1_sc}\n"
                f"\tF1.5 = {f1_sc}\n")

    return f1_5


def _init_and_train_model(architecture, train_data, validation, window_size, lr, steps_per_epoch, epochs) -> (
        AutoEncoder | AdversarialAutoEncoder, History):
    if architecture == "ae":
        model: AutoEncoder = AutoEncoder(window_size)
        model.build(train_data.shape)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='mse')

    elif architecture == "aae":
        model: AdversarialAutoEncoder = AdversarialAutoEncoder(window_size)
        model.build(train_data.shape)

        autoencoder_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(discriminator_optimizer, autoencoder_optimizer)
    else:
        raise TypeError(f"Unknown architecture: {architecture}")

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    training_history: History

    if validation:
        train_normal, val_normal = train_test_split(train_data, test_size=0.2)

        train_dataset = tf.data.Dataset.from_tensor_slices((train_normal, train_normal)).repeat().batch(256)
        validation_dataset = tf.data.Dataset.from_tensor_slices((val_normal, val_normal)).batch(256)

        training_history = model.fit(train_dataset, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                     callbacks=[callback, tensorboard_callback],
                                     validation_data=validation_dataset)
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_data)).repeat().batch(256)
        training_history = model.fit(train_dataset, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                     callbacks=[callback, tensorboard_callback])

    # model.summary()

    return model, training_history
