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
    def prepare_experiment_data(df_kuka_normal, df_kuka_slow, window_size, window_step_size):
        normal_data_windowed = DataUtils.sliding_window(df_kuka_normal, window_size, window_step_size)
        test_data_slow = DataUtils.sliding_window(df_kuka_slow, window_size, window_step_size)

        train_data, test_data_normal = train_test_split(normal_data_windowed, test_size=0.3)

        test_data = np.concatenate([test_data_slow, test_data_normal])
        test_data_labels = np.concatenate([
            np.ones(test_data_slow.shape[0]),  # Anomalies
            np.zeros(test_data_normal.shape[0]),  # Normal
        ])

        print(f"Training data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}, "
              f"formed by: {test_data_slow.shape} anomalies and {test_data_normal.shape} normal")

        shuffled_indices = np.random.permutation(len(test_data))
        test_data_shuffled_tensor = tf.convert_to_tensor(test_data[shuffled_indices])
        test_data_labels_shuffled = test_data_labels[shuffled_indices]
        test_data_slow_tensor = tf.convert_to_tensor(test_data_slow)

        return train_data, test_data_shuffled_tensor, test_data_labels_shuffled, test_data_slow_tensor
