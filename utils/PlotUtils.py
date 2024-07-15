import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, \
    ConfusionMatrixDisplay, roc_curve, roc_auc_score

from utils.DataUtils import DataUtils


class PlotUtils:

    @staticmethod
    def plot_reconstruction_error(train_data, reconstructions, config):
        sample_inside_window = 4  # from 0 to 10
        window_number = 20  # from 0 to the number of windows created

        # Plot the reconstruction error for each feature
        plt.plot(train_data[window_number, :, sample_inside_window], 'b')
        plt.plot(reconstructions[window_number, :, sample_inside_window], 'r')
        plt.fill_between(np.arange(86), reconstructions[window_number, :, sample_inside_window],
                         train_data[window_number, :, sample_inside_window], color='lightcoral')
        plt.legend(labels=["Input", "Reconstruction", "Error"])

        if not os.path.exists(f"{config['ARCHITECTURE']}/plot_reconstr"):
            os.makedirs(f"{config['ARCHITECTURE']}/plot_reconstr")
        plt.savefig(f"{config['ARCHITECTURE']}/plot_reconstr/ws{config['WINDOW_SIZE']}_lr{config['LR']}_ep{config['EPOCHS']}.png")

        plt.show()

    @staticmethod
    def plot_averaged_reconstruction_error(train_data, reconstructions, config):
        train_data_averaged_over_samples = np.mean(np.mean(train_data, axis=2), axis=0)
        reconstructions_over_samples = np.mean(np.mean(reconstructions, axis=2), axis=0)

        # Plot the reconstruction error for each feature
        plt.plot(train_data_averaged_over_samples, 'b')
        plt.plot(reconstructions_over_samples, 'r')
        plt.fill_between(np.arange(86), reconstructions_over_samples,
                         train_data_averaged_over_samples, color='lightcoral')
        plt.legend(labels=["Input", "Reconstruction", "Error"])

        if not os.path.exists(f"{config['ARCHITECTURE']}/plot_reconstr"):
            os.makedirs(f"{config['ARCHITECTURE']}/plot_reconstr")
        plt.savefig(
            f"{config['ARCHITECTURE']}/plot_reconstr/ws{config['WINDOW_SIZE']}_lr{config['LR']}_ep{config['EPOCHS']}.png")

        plt.show()

    @staticmethod
    def plot_loss(mean_mse):
        plt.hist(mean_mse[:, None], bins=20)
        plt.xlabel("Loss")
        plt.ylabel("No of examples")
        plt.show()

    @staticmethod
    def plot_train_and_test_losses(mean_mse_train, mean_mse_test, threshold, config):
        plt.hist(mean_mse_train[:, None], bins=40, color="b")
        plt.hist(mean_mse_test[:, None], bins=40, color="r")
        plt.axvline(x=threshold, color="green", linestyle="--")
        plt.xlabel("Loss")
        plt.ylabel("No of examples")

        if not os.path.exists(f"{config['ARCHITECTURE']}/plot_loss"):
            os.makedirs(f"{config['ARCHITECTURE']}/plot_loss")
        plt.savefig(f"{config['ARCHITECTURE']}/plot_loss/ws{config['WINDOW_SIZE']}_lr{config['LR']}_ep{config['EPOCHS']}.png")

        plt.show()

    @staticmethod
    def plot_train_and_test_split_losses(mean_mse_train, mean_mse_normal_test, mean_mse_anomaly_test, threshold, config):
        plt.hist(mean_mse_train[:, None], bins=40, color="b")
        plt.hist(mean_mse_normal_test[:, None], bins=40, color="g")
        plt.hist(mean_mse_anomaly_test[:, None], bins=40, color="r")
        plt.axvline(x=threshold, color="black", linestyle="--")
        plt.xlabel("Loss")
        plt.ylabel("No of examples")

        if not os.path.exists(f"{config['ARCHITECTURE']}/plot_loss"):
            os.makedirs(f"{config['ARCHITECTURE']}/plot_loss")
        plt.savefig(
            f"{config['ARCHITECTURE']}/plot_loss/ws{config['WINDOW_SIZE']}_lr{config['LR']}_ep{config['EPOCHS']}.png")

        plt.show()

    @staticmethod
    def plot_train_and_test_losses_over_features(mse_train, mean_mse_normal_test, mean_mse_anomaly_test, threshold_vector, config):
        def _plot_custom_histogram(data, threshold, ax):
            mean_mse_train, mean_mse_normal_test, mean_mse_anomaly_test = data
            ax.hist(mean_mse_train[:, None], bins=40, color="b", alpha=0.7, label='Train')
            ax.hist(mean_mse_normal_test[:, None], bins=40, color="g", alpha=0.7, label='Test Normal')
            ax.hist(mean_mse_anomaly_test[:, None], bins=40, color="r", alpha=0.7, label='Test Anomalies')
            ax.axvline(x=threshold, color="black", linestyle="--", label='Threshold')
            ax.set_xlabel("Loss")
            ax.set_ylabel("No of examples")
            ax.legend()

        df_train = pd.DataFrame(mse_train[:, :20])
        df_test_normal = pd.DataFrame(mean_mse_normal_test[:, :20])
        df_test_anomalies = pd.DataFrame(mean_mse_anomaly_test[:, :20])

        df_melted = df_train.melt(var_name='Feature', value_name='Value')
        df_melted_test_normal = df_test_normal.melt(var_name='Feature', value_name='Value')
        df_melted_test_anomalies = df_test_anomalies.melt(var_name='Feature', value_name='Value')

        # Combine train and test data in one DataFrame
        df_combined = pd.DataFrame({
            'Feature': df_melted['Feature'],
            'Train': df_melted['Value'],
            'Test Normal': df_melted_test_normal['Value'],
            'Test Anomalies': df_melted_test_anomalies['Value']
        })

        # Create a FacetGrid
        g = sns.FacetGrid(df_combined, col='Feature', col_wrap=5, height=2, aspect=1.5, sharex=False, sharey=False)

        # Map the custom plotting function to each subplot
        for threshold, ax, feature in zip(threshold_vector, g.axes.flat, df_combined['Feature'].unique()):
            _plot_custom_histogram((df_train[feature].values, df_test_normal[feature].values, df_test_anomalies[feature].values), threshold, ax)

        # Adjust the layout and show the plot
        plt.tight_layout()

        if not os.path.exists(f"{config['ARCHITECTURE']}/plot_loss_vector"):
            os.makedirs(f"{config['ARCHITECTURE']}/plot_loss_vector")
        plt.savefig(f"{config['ARCHITECTURE']}/plot_loss_vector/ws{config['WINDOW_SIZE']}_lr{config['LR']}_ep{config['EPOCHS']}.png")

        plt.show()

    @staticmethod
    def plot_confusion_matrix(predictions, labels, config):
        # False Positive: Normal data classified as anomaly
        # False Negative: Anomaly data classified as normal
        cm = confusion_matrix(labels, predictions.numpy(), labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot()

        if not os.path.exists(f"{config['ARCHITECTURE']}/conf_matrix"):
            os.makedirs(f"{config['ARCHITECTURE']}/conf_matrix")
        plt.savefig(f"{config['ARCHITECTURE']}/conf_matrix/ws{config['WINDOW_SIZE']}_lr{config['LR']}_ep{config['EPOCHS']}.png")

        plt.show()

    @staticmethod
    def plot_roc_curve(predictions, labels, config):
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

        if not os.path.exists(f"{config['ARCHITECTURE']}/roc_curve"):
            os.makedirs(f"{config['ARCHITECTURE']}/roc_curve")
        plt.savefig(f"{config['ARCHITECTURE']}/roc_curve/ws{config['WINDOW_SIZE']}_lr{config['LR']}_ep{config['EPOCHS']}.png")

        plt.show()
        plt.close()

    
    @staticmethod
    def plot_execution_logs():
        current_dir = os.getcwd()
        ae_df = DataUtils.parse_log_file(os.path.join(current_dir, 'AE_logging_file.txt'))
        aae_df = DataUtils.parse_log_file(os.path.join(current_dir,'AAE_logging_file.txt'))

        # Add a column to distinguish between AE and AAE
        ae_df['Architecture'] = 'AE'
        aae_df['Architecture'] = 'AAE'

    #       # Combine both DataFrames
        df = pd.concat([ae_df, aae_df], ignore_index=True)
        
    # def _inner_plot_execution_logs(df):
        sns.set(style="whitegrid")
        plt.figure(figsize=(20, 15))

        # Find the rows with the best scores for AE and AAE separately
        row_best_f1_ae = df[df['Architecture'] == 'AE'].loc[df[df['Architecture'] == 'AE']['F1 Score'].idxmax()]
        row_best_f1_aae = df[df['Architecture'] == 'AAE'].loc[df[df['Architecture'] == 'AAE']['F1 Score'].idxmax()]

        row_best_f1_5_ae = df[df['Architecture'] == 'AE'].loc[df[df['Architecture'] == 'AE']['F1.5 Score'].idxmax()]
        row_best_f1_5_aae = df[df['Architecture'] == 'AAE'].loc[df[df['Architecture'] == 'AAE']['F1.5 Score'].idxmax()]

        # Plot 1: Effect of Window Size on F1 Score
        plt.subplot(2, 2, 1)
        sns.lineplot(data=df[
            ((df['Learning Rate'] == row_best_f1_ae['Learning Rate']) & (df['Epochs'] == row_best_f1_ae['Epochs']) & (df['Architecture'] == 'AE')) |
            ((df['Learning Rate'] == row_best_f1_aae['Learning Rate']) & (df['Epochs'] == row_best_f1_aae['Epochs']) & (df['Architecture'] == 'AAE'))],
                    x='Window Size', y='F1 Score', hue='Architecture', markers=True)
        plt.title(
            f'Effect of Window Size on F1 Score\nAE LR = {row_best_f1_ae["Learning Rate"]}, Epochs = {row_best_f1_ae["Epochs"]}\nAAE LR = {row_best_f1_aae["Learning Rate"]}, Epochs = {row_best_f1_aae["Epochs"]}')
        plt.ylim(0, 1)
        plt.legend(title='Architecture')

        # Plot 2: Effect of Epochs on F1 Score
        plt.subplot(2, 2, 2)
        sns.lineplot(data=df[
            ((df['Window Size'] == row_best_f1_ae['Window Size']) & (df['Learning Rate'] == row_best_f1_ae['Learning Rate']) & (df['Architecture'] == 'AE')) |
            ((df['Window Size'] == row_best_f1_aae['Window Size']) & (df['Learning Rate'] == row_best_f1_aae['Learning Rate']) & (df['Architecture'] == 'AAE'))],
                    x='Epochs', y='F1 Score', hue='Architecture', markers=True)
        plt.title(
            f"Effect of Epochs on F1 Score\nAE WS = {row_best_f1_ae['Window Size']}, LR = {row_best_f1_ae['Learning Rate']}\nAAE WS = {row_best_f1_aae['Window Size']}, LR = {row_best_f1_aae['Learning Rate']}")
        plt.ylim(0.75, 1)
        plt.legend(title='Architecture')

        # Plot 3: Effect of Window Size on F1.5 Score
        plt.subplot(2, 2, 3)
        sns.lineplot(data=df[
            ((df['Learning Rate'] == row_best_f1_5_ae['Learning Rate']) & (df['Epochs'] == row_best_f1_5_ae['Epochs']) & (df['Architecture'] == 'AE')) |
            ((df['Learning Rate'] == row_best_f1_5_aae['Learning Rate']) & (df['Epochs'] == row_best_f1_5_aae['Epochs']) & (df['Architecture'] == 'AAE'))],
                    x='Window Size', y='F1.5 Score', hue='Architecture', markers=True)
        plt.title(
            f'Effect of Window Size on F1.5 Score\nAE LR = {row_best_f1_ae["Learning Rate"]}, Epochs = {row_best_f1_ae["Epochs"]}\nAAE LR = {row_best_f1_aae["Learning Rate"]}, Epochs = {row_best_f1_aae["Epochs"]}')
        plt.ylim(0, 1)
        plt.legend(title='Architecture')
        
        # Plot 4: Effect of Epochs on F1.5 Score
        plt.subplot(2, 2, 4)
        sns.lineplot(data=df[
            ((df['Window Size'] == row_best_f1_5_ae['Window Size']) & (df['Learning Rate'] == row_best_f1_5_ae['Learning Rate']) & (df['Architecture'] == 'AE')) |
            ((df['Window Size'] == row_best_f1_5_aae['Window Size']) & (df['Learning Rate'] == row_best_f1_5_aae['Learning Rate']) & (df['Architecture'] == 'AAE'))],
                    x='Epochs', y='F1.5 Score', hue='Architecture', markers=True)
        plt.title(
            f'Effect of Epochs on F1.5 Score\nAE WS = {row_best_f1_ae["Window Size"]}, LR = {row_best_f1_ae["Learning Rate"]}\nAAE WS = {row_best_f1_aae["Window Size"]}, LR = {row_best_f1_aae["Learning Rate"]}')
        plt.ylim(0.75, 1)
        plt.legend(title='Architecture')

        plt.tight_layout()
        plt.savefig('combined_analysis.png')
        plt.close()

        plt.show()
        print("Combined analysis complete. Plot saved as 'combined_analysis.png'.")
        print(f"Best F1 AE: \n{row_best_f1_ae}\n")
        print(f"Best F1 AAE: \n{row_best_f1_aae}\n")

        print(f"Best F1.5 AE: \n{row_best_f1_5_ae}\n")
        print(f"Best F1.5 AAE: \n{row_best_f1_5_aae}\n")
