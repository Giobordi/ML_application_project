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

BATCH_SIZE = 256


class ExperimentUtils:

    @staticmethod
    def ae_experiment(df_kuka_normal, df_kuka_slow, window_size, window_step_size, lr, epochs,
                      validation: bool, mean_threshold: bool):
        print(f"-----------------------------------------------------------------------------------")
        print(f"Running AE experiment with config: WS = {window_size}, lr = {lr}, epochs = {epochs}")
        print(f"-----------------------------------------------------------------------------------")

        architecture = "ae"
        data = DataUtils.prepare_experiment_data(df_kuka_normal, df_kuka_slow, window_size, window_step_size)
        train_data, test_data_shuffled_tensor, test_data_labels_shuffled, test_data_slow_tensor = data

        # AutoEncoder
        autoencoder, train_history_ae = _init_and_train_model(
            architecture, train_data, validation, window_size, lr, epochs
        )
        print(train_history_ae.history)

        config = {
            "ARCHITECTURE": architecture,
            "EPOCHS": epochs,
            "LR": lr,
            "WINDOW_SIZE": window_size,
            "MEAN_THRESHOLD": mean_threshold
        }

        return ExperimentUtils.run_experiment(autoencoder, train_data, test_data_slow_tensor, test_data_shuffled_tensor,
                                              test_data_labels_shuffled, config)

    @staticmethod
    def aae_experiment(df_kuka_normal, df_kuka_slow, window_size, window_step_size, lr, epochs,
                       validation: bool, mean_threshold: bool):
        print(f"-----------------------------------------------------------------------------------")
        print(f"Running AAE experiment with config: WS = {window_size}, lr = {lr}, epochs = {epochs}")
        print(f"-----------------------------------------------------------------------------------")

        architecture = "aae"
        data = DataUtils.prepare_experiment_data(df_kuka_normal, df_kuka_slow, window_size, window_step_size)
        train_data, test_data_shuffled_tensor, test_data_labels_shuffled, test_data_slow_tensor = data

        # Adversarial AutoEncoder
        aae, train_history_aae = _init_and_train_model(
            architecture, train_data, validation, window_size, lr, epochs
        )
        print(train_history_aae.history)

        config = {
            "ARCHITECTURE": architecture,
            "EPOCHS": epochs,
            "LR": lr,
            "WINDOW_SIZE": window_size,
            "MEAN_THRESHOLD": mean_threshold
        }

        return ExperimentUtils.run_experiment(aae, train_data, test_data_slow_tensor, test_data_shuffled_tensor,
                                              test_data_labels_shuffled, config)

    @staticmethod
    def run_experiment(model, train_data, test_data_slow_tensor, test_data_shuffled_tensor, test_data_labels_shuffled,
                       config):

        normal_reconstructions = model.predict(train_data)
        PlotUtils.plot_averaged_reconstruction_error(train_data, normal_reconstructions, config)

        normal_train_loss = tf.keras.losses.MeanSquaredError().call(train_data, normal_reconstructions)

        if config['MEAN_THRESHOLD']:
            mean_mse = np.mean(normal_train_loss.numpy(), axis=1)
            # plot_loss(mean_mse)
            threshold = np.mean(mean_mse) + 3 * np.std(mean_mse)

            test_reconstructions = model.predict(test_data_shuffled_tensor)
            test_loss = tf.keras.losses.MeanSquaredError().call(test_data_shuffled_tensor, test_reconstructions)

            normal_test_losses_indices = np.where(test_data_labels_shuffled == 0)[0]
            anomalies_test_losses_indices = np.where(test_data_labels_shuffled == 1)[0]

            normal_losses = tf.gather(test_loss, normal_test_losses_indices)
            anomaly_losses = tf.gather(test_loss, anomalies_test_losses_indices)

            mean_mse_normal_test = np.mean(normal_losses.numpy(), axis=1)
            mean_mse_anomaly_test = np.mean(anomaly_losses.numpy(), axis=1)

            PlotUtils.plot_train_and_test_split_losses(mean_mse, mean_mse_normal_test, mean_mse_anomaly_test, threshold, config)

            preds = _classify(test_loss, threshold)
        else:
            mse = normal_train_loss.numpy()
            threshold_vector = np.mean(mse, axis=0) + 3 * np.std(mse, axis=0)

            # TODO: plot_train_and_test_split_losses

            test_reconstructions_slow = model.predict(test_data_slow_tensor)
            test_loss_slow = tf.keras.losses.MeanSquaredError().call(test_data_slow_tensor, test_reconstructions_slow)
            mse_test_slow = test_loss_slow.numpy()

            PlotUtils.plot_train_and_test_losses_over_features(mse, mse_test_slow, threshold_vector, config)

            preds = _predict_threshold_vector(model, test_data_shuffled_tensor, threshold_vector)

        f1_5 = _print_stats(preds, test_data_labels_shuffled, config)

        PlotUtils.plot_confusion_matrix(preds, test_data_labels_shuffled, config)
        PlotUtils.plot_roc_curve(preds, test_data_labels_shuffled, config)

        return f1_5


def _classify(loss, threshold):
    # loss: (ns, 86)
    mean_mse_test = np.mean(loss.numpy(), axis=1)
    # mean_mse_test: (ns,)
    # thr: 1
    # out: (ns,)
    return tf.math.greater(mean_mse_test, threshold)


def _predict_threshold_vector(model, data, threshold):
    reconstructions = model.predict(data)
    loss = tf.keras.losses.MeanSquaredError().call(data, reconstructions)
    mse_test = loss.numpy()
    # mse_test: (ns, 86)
    # thr: (86,)
    # out: (ns,)
    return tf.math.reduce_all(tf.math.greater(mse_test, threshold), axis=1)


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
                f"\tF1.5 = {f1_5}\n")

    return f1_5


def _init_and_train_model(architecture, train_data, validation, window_size, lr, epochs) -> (
        AutoEncoder | AdversarialAutoEncoder, History):
    if architecture == "ae":
        model: AutoEncoder = AutoEncoder(window_size)
        model.build(train_data.shape)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='mse')
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    elif architecture == "aae":
        model: AdversarialAutoEncoder = AdversarialAutoEncoder(window_size)
        model.build(train_data.shape)

        autoencoder_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(discriminator_optimizer, autoencoder_optimizer)
        callback = None
    else:
        raise TypeError(f"Unknown architecture: {architecture}")

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    training_history: History

    if validation:
        train_normal, val_normal = train_test_split(train_data, test_size=0.2)

        train_dataset = (tf.data.Dataset.from_tensor_slices((train_normal, train_normal))
                         .repeat()
                         .shuffle(buffer_size=len(train_normal))
                         .batch(BATCH_SIZE))
        validation_dataset = tf.data.Dataset.from_tensor_slices((val_normal, val_normal)).batch(BATCH_SIZE)

        steps_per_epoch = int(np.ceil(train_normal.shape[0] / BATCH_SIZE))
        training_history = model.fit(train_dataset, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                     callbacks=[cb for cb in [callback, tensorboard_callback] if cb is not None],
                                     validation_data=validation_dataset)
    else:
        train_dataset = (tf.data.Dataset.from_tensor_slices((train_data, train_data))
                         .repeat()
                         .shuffle(buffer_size=len(train_data))
                         .batch(BATCH_SIZE))

        steps_per_epoch = int(np.ceil(train_data.shape[0] / BATCH_SIZE))
        training_history = model.fit(train_dataset, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                     callbacks=[cb for cb in [callback, tensorboard_callback] if cb is not None])

    # model.summary()

    return model, training_history
