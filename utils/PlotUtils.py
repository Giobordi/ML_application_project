import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, \
    ConfusionMatrixDisplay, roc_curve, roc_auc_score


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
