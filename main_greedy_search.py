from utils.DataUtils import DataUtils
from utils.ExperimentUtils import ExperimentUtils


def main():
    # Default config
    lr = 1e-3
    epochs = 2
    steps_per_epoch = 100
    validation = False

    default_config = (lr, epochs, steps_per_epoch, validation)

    # Load dataset
    df_kuka_normal, df_kuka_slow = DataUtils.load_dataset()

    architecture = "ae"
    greedy_search(architecture, df_kuka_normal, df_kuka_slow, default_config)

    architecture = "aae"
    greedy_search(architecture, df_kuka_normal, df_kuka_slow, default_config)


def greedy_search(architecture, df_kuka_normal, df_kuka_slow, default_config):
    if architecture == "ae":
        experiment = ExperimentUtils.ae_experiment
    elif architecture == "aae":
        experiment = ExperimentUtils.aae_experiment
    else:
        raise TypeError(f"Invalid architecture {architecture}")

    # Load default config
    lr, epochs, steps_per_epoch, validation = default_config

    # -----------------------------
    # - Search best window config -
    # -----------------------------
    best_f1_5 = None
    best_window_config = None

    for window_size, window_step_size in zip([10, 20, 30, 50, 100, 200], [1, 2, 3, 5, 10, 20]):
        f1_5 = experiment(df_kuka_normal, df_kuka_slow, window_size, window_step_size, lr,
                          steps_per_epoch, epochs, validation)

        if best_f1_5 is None or f1_5 > best_f1_5:
            best_f1_5 = f1_5
            best_window_config = (window_size, window_step_size)

    # Fix window config
    final_window_size, final_window_step_size = best_window_config

    # ------------------------------------
    # - Search best learning rate config -
    # ------------------------------------
    best_f1_5 = None
    best_lr_config = None
    for lr in [1e-3, 1e-4, 1e-5, 1e-2]:
        f1_5 = experiment(df_kuka_normal, df_kuka_slow, final_window_size, final_window_step_size,
                          lr, steps_per_epoch, epochs, validation)

        if best_f1_5 is None or f1_5 > best_f1_5:
            best_f1_5 = f1_5
            best_lr_config = lr

    # Fix window config
    final_lr = best_lr_config

    # -------------------------------------
    # - Search best training epoch config -
    # -------------------------------------
    best_f1_5 = None
    best_epochs_config = None
    for epochs in [2, 5, 10]:
        f1_5 = experiment(df_kuka_normal, df_kuka_slow, final_window_size, final_window_step_size,
                          final_lr, steps_per_epoch, epochs, validation)

        if best_f1_5 is None or f1_5 > best_f1_5:
            best_f1_5 = f1_5
            best_epochs_config = lr

    final_epochs = best_epochs_config

    # ----------------
    # - Final config -
    # ----------------
    print(f"Best configuration for AE gives F1.5 = {best_f1_5}, with: \n"
          f"\t- Window Size = {final_window_size},\n"
          f"\t- Window Step Size = {final_window_step_size}\n"
          f"\t- Epochs = {final_epochs},\n"
          f"\t- LR = {final_lr}\n")


if __name__ == '__main__':
    main()
