import os

from dotenv import load_dotenv

from utils.DataUtils import DataUtils
from utils.ExperimentUtils import ExperimentUtils

load_dotenv(".env")
WINDOW_SIZE = 10 
LR = 0.001
EPOCHS = 100
ARCHITECTURE = "aae"
WINDOW_STEP_SIZE = 1


def main():
    global WINDOW_SIZE, LR, EPOCHS, ARCHITECTURE, WINDOW_STEP_SIZE
    # Load config
    window_size = int(os.getenv("WINDOW_SIZE"))
    window_step_size = int(os.getenv("WINDOW_STEP_SIZE"))
    lr = float(os.getenv("LEARNING_RATE"))
    epochs = int(os.getenv("EPOCHS"))
    validation = os.getenv("VALIDATION").strip() == "True"
    mean_threshold = os.getenv("MEAN_THRESHOLD").strip() == "True"
    
    # Load dataset
    df_kuka_normal, df_kuka_slow = DataUtils.load_dataset()

    best_ae_f1_5 = best_ae_config = None
    best_aae_f1_5 = best_aae_config = None

    # for window_size in [10, 20, 30, 50, 100, 200]:
    for window_size in [200]:
        window_step_size = window_size // 4

        for lr in [1e-3, 1e-4, 1e-5, 1e-2]:
            LR = lr
            for epochs in [2, 5, 10]:
                EPOCHS = epochs

                # AE
                ae_f1_5 = ExperimentUtils.ae_experiment(df_kuka_normal, df_kuka_slow, window_size,
                                                        window_step_size, lr, epochs, validation, mean_threshold)

                if best_ae_f1_5 is None or ae_f1_5 > best_ae_f1_5:
                    best_ae_f1_5 = ae_f1_5
                    best_ae_config = (window_size, lr, epochs)

                # AAE
                aae_f1_5 = ExperimentUtils.aae_experiment(df_kuka_normal, df_kuka_slow, window_size,
                                                          window_step_size, lr, epochs, validation, mean_threshold)

                if best_aae_f1_5 is None or aae_f1_5 > best_aae_f1_5:
                    best_aae_f1_5 = aae_f1_5
                    best_aae_config = (window_size, lr, epochs)


if __name__ == '__main__':
    main()
