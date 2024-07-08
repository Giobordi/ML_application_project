import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from utils.DataUtils import DataUtils
from utils.ExperimentUtils import ExperimentUtils
from utils.PlotUtils import PlotUtils

load_dotenv(".env")


# Anomaly with label 1
# Normal with label 0

def main():
    # Load config
    window_size = int(os.getenv("WINDOW_SIZE"))
    window_step_size = int(os.getenv("WINDOW_STEP_SIZE"))
    lr = float(os.getenv("LEARNING_RATE"))
    epochs = int(os.getenv("EPOCHS"))
    validation = os.getenv("VALIDATION").strip() == "True"
    mean_threshold = os.getenv("MEAN_THRESHOLD").strip() == "True"

    # Load dataset
    # df_kuka_normal, df_kuka_slow = DataUtils.load_dataset()
    # ExperimentUtils.ae_experiment(df_kuka_normal, df_kuka_slow, window_size, window_step_size, lr, epochs, validation, mean_threshold)
    # ExperimentUtils.aae_experiment(df_kuka_normal, df_kuka_slow, window_size, window_step_size, lr, epochs, validation, mean_threshold)

    PlotUtils.plot_execution_logs()


if __name__ == '__main__':
    main()
