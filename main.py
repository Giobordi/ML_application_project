import os

from dotenv import load_dotenv

from utils.DataUtils import DataUtils
from utils.ExpermentUtils import ExperimentUtils

load_dotenv(".env")


# Anomaly with label 1
# Normal with label 0

def main():
    # Load config
    window_size = int(os.getenv("WINDOW_SIZE"))
    window_step_size = int(os.getenv("WINDOW_STEP_SIZE"))
    lr = float(os.getenv("LEARNING_RATE"))
    steps_per_epoch = int(os.getenv("STEPS_PER_EPOCH"))
    epochs = int(os.getenv("EPOCHS"))
    validation = os.getenv("VALIDATION").strip() == "True"

    # Load dataset
    df_kuka_normal, df_kuka_slow = DataUtils.load_dataset()
    ExperimentUtils.ae_experiment(df_kuka_normal, df_kuka_slow, window_size, window_step_size, lr, steps_per_epoch, epochs, validation)
    ExperimentUtils.aae_experiment(df_kuka_normal, df_kuka_slow, window_size, window_step_size, lr, steps_per_epoch, epochs, validation)


if __name__ == '__main__':
    main()
