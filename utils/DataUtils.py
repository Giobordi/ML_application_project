import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


class DataUtils:

    @staticmethod
    def load_dataset():
        root_dir_dataset = os.getenv("ROOTDIR_DATASET")

        # Load normal dataset
        kuka_column_names_path = os.path.join(root_dir_dataset, 'KukaColumnNames.npy')
        kuka_column_names = np.load(kuka_column_names_path)
        kuka_column_names.reshape((1, -1))
        # Drop "anomaly" column
        kuka_column_names = kuka_column_names[:-1]

        kuka_normal_path = os.path.join(root_dir_dataset, 'KukaNormal.npy')
        kuka_normal = np.load(kuka_normal_path)
        kuka_normal = kuka_normal.astype('float32')
        denominator_normal = np.where((np.max(kuka_normal, axis=0) - np.min(kuka_normal, axis=0)) == 0, 1,
                                      (np.max(kuka_normal, axis=0) - np.min(kuka_normal, axis=0)))
        kuka_normal = (kuka_normal - np.min(kuka_normal, axis=0)) / denominator_normal

        kuka_normal = np.vstack((kuka_column_names, kuka_normal))
        df_kuka_normal = pd.DataFrame(kuka_normal)
        df_kuka_normal.columns = df_kuka_normal.iloc[0]
        df_kuka_normal = df_kuka_normal.iloc[1:]

        df_kuka_normal = df_kuka_normal.reset_index(drop=True)

        # Load anomalies
        kuka_slow_path = os.path.join(root_dir_dataset, 'KukaSlow.npy')
        kuka_slow = np.load(kuka_slow_path)
        # Drop "anomaly" column
        kuka_slow = kuka_slow[:, :-1]
        kuka_slow = kuka_slow.astype('float32')
        denominator_slow = np.where((np.max(kuka_slow, axis=0) - np.min(kuka_slow, axis=0)) == 0, 1,
                                    (np.max(kuka_slow, axis=0) - np.min(kuka_slow, axis=0)))
        kuka_slow = (kuka_slow - np.min(kuka_slow, axis=0)) / denominator_slow

        kuka_slow = np.vstack((kuka_column_names, kuka_slow))

        df_kuka_slow = pd.DataFrame(kuka_slow)
        df_kuka_slow.columns = df_kuka_slow.iloc[0]
        df_kuka_slow = df_kuka_slow.iloc[1:]
        df_kuka_slow = df_kuka_slow.reset_index(drop=True)

        df_kuka_normal = df_kuka_normal.astype('float32')
        df_kuka_slow = df_kuka_slow.astype('float32')

        return df_kuka_normal, df_kuka_slow

    @staticmethod
    def sliding_window(data, window_size, step_size):
        windows = []
        for i in range(0, data.shape[0] - window_size + 1, step_size):
            windows.append(data.iloc[i:i + window_size].to_numpy().astype('float32'))

        x = np.array(windows).astype('float32')
        return np.swapaxes(x, 1, 2)

    @staticmethod
    def sliding_window_with_labels(data, window_size, step_size):
        windows = []
        window_labels = []
        for i in range(0, data.shape[0] - window_size + 1, step_size):
            window = data.iloc[i:i + window_size].copy()
            if window[window['anomaly'] == 1].shape[0] > 0:
                label = 1
            else:
                label = 0
            window.drop('anomaly', axis=1, inplace=True)

            window_labels.append(label)
            windows.append(window.to_numpy().astype('float32'))

        x = np.array(windows).astype('float32')
        y = np.array(window_labels).astype('float32')
        return np.swapaxes(x, 1, 2), y

    @staticmethod
    def prepare_experiment_data(df_kuka_normal, df_kuka_slow, window_size, window_step_size):
        normal_data_windowed = DataUtils.sliding_window(df_kuka_normal, window_size, window_step_size)
        test_data_slow = DataUtils.sliding_window(df_kuka_slow, window_size, window_step_size)

        mixed_windowed_dataset, mixed_windowed_dataset_labels = _insert_anomaly_in_normal_task(df_kuka_normal, df_kuka_slow, window_size, window_step_size)

        train_data, test_data_normal = train_test_split(normal_data_windowed, test_size=0.3)

        test_data = np.concatenate([test_data_slow, test_data_normal, mixed_windowed_dataset])
        test_data_labels = np.concatenate([
            np.ones(test_data_slow.shape[0]),  # Anomalies
            np.zeros(test_data_normal.shape[0]),  # Normal
            mixed_windowed_dataset_labels   # Mixed
        ])

        print(f"Training data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}, "
              f"formed by: {test_data_slow.shape} anomalies, {test_data_normal.shape} normal and {mixed_windowed_dataset.shape} mixed")

        shuffled_indices = np.random.permutation(len(test_data))
        test_data_shuffled_tensor = tf.convert_to_tensor(test_data[shuffled_indices])
        test_data_labels_shuffled = test_data_labels[shuffled_indices]
        test_data_slow_tensor = tf.convert_to_tensor(test_data_slow)

        return train_data, test_data_shuffled_tensor, test_data_labels_shuffled, test_data_slow_tensor


def _insert_anomaly_in_normal_task(normal_array, anomaly_array, window_size, window_step_size):
    _, normal_subset = train_test_split(normal_array, test_size=0.3, shuffle=False)
    anomaly_copy = anomaly_array.copy()

    normal_subset.reset_index(drop=True, inplace=True)

    normal_subset['anomaly'] = 0
    anomaly_copy['anomaly'] = 1

    num_anomalies_to_insert = window_size // 10
    windows = np.arange(0, normal_subset.shape[0] - window_size, window_size)

    # Generate random indices for all windows at once
    random_indices = np.concatenate([
        np.random.choice(range(i, i + window_size), size=num_anomalies_to_insert, replace=False)
        for i in windows
    ])

    # For each action, select anomalies with the same action
    for action in normal_subset.iloc[random_indices]['action'].unique():
        anomalies_with_same_action = anomaly_copy[anomaly_copy['action'] == action]
        normal_with_same_action = normal_subset.iloc[random_indices][normal_subset.iloc[random_indices]['action'] == action]

        anomalies_chosen_indices = np.random.choice(anomalies_with_same_action.index,
                                                    size=np.min((
                                                        anomalies_with_same_action.shape[0],
                                                        normal_with_same_action.shape[0]
                                                    )),
                                                    replace=False)

        if normal_with_same_action.shape[0] > anomalies_chosen_indices.shape[0]:
            normal_indices_to_replace = np.random.choice(normal_with_same_action.index, size=anomalies_chosen_indices.shape[0], replace=False)
        else:
            normal_indices_to_replace = normal_with_same_action.index

        # Replace normal data with anomalies
        normal_subset.loc[normal_indices_to_replace] = anomalies_with_same_action.loc[anomalies_chosen_indices].values

    mixed_windowed_dataset, mixed_windowed_dataset_labels = DataUtils.sliding_window_with_labels(normal_subset, window_size, window_step_size)

    return mixed_windowed_dataset, mixed_windowed_dataset_labels

