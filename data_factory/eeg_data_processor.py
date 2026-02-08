import numpy as np
import random


class ECGDataProcessor:
    def __init__(self, dataset_path="./DeepSeparator/data/"):
        self.dataset_path = dataset_path
        self.train_num = 3000
        self.test_num = 400

        self.EEG_all = np.load(dataset_path + "EEG_all_epochs.npy")
        self.EOG_all = np.load(dataset_path + "EOG_all_epochs.npy")
        self.EMG_all = np.load(dataset_path + "EMG_all_epochs.npy")

        self.snr_levels = [-4, -2, 0, 2, 4]

    def preprocess(self):
        EEG_train = self.EEG_all[: self.train_num]
        EEG_test = self.EEG_all[self.train_num : self.train_num + self.test_num]

        EOG_train = self.EOG_all[: self.train_num]
        EOG_test = self.EOG_all[self.train_num : self.train_num + self.test_num]

        EMG_train = self.EMG_all[: self.train_num]
        EMG_test = self.EMG_all[self.train_num : self.train_num + self.test_num]

        EEG_train_power = np.mean(EEG_train**2)
        EOG_train_power = np.mean(EOG_train**2)
        EMG_train_power = np.mean(EMG_train**2)
        EOG_EMG_train_power = np.mean((EOG_train + EMG_train) ** 2)

        EEG_test_power = np.mean(EEG_test**2)
        EOG_test_power = np.mean(EOG_test**2)
        EMG_test_power = np.mean(EMG_test**2)
        EOG_EMG_test_power = np.mean((EOG_test + EMG_test) ** 2)

        for snr in self.snr_levels:
            EOG_train_scale = self.scale(EEG_train_power, EOG_train_power, snr)
            EMG_train_scale = self.scale(EEG_train_power, EMG_train_power, snr)
            EOG_EMG_train_scale = self.scale(EEG_train_power, EOG_EMG_train_power, snr)

            noisy_EOG_train = EEG_train + EOG_train * EOG_train_scale
            noisy_EMG_train = EEG_train + EMG_train * EMG_train_scale
            noisy_EOGEMG_train = (
                EEG_train + (EOG_train + EMG_train) * EOG_EMG_train_scale
            )

            EOG_test_scale = self.scale(EEG_test_power, EOG_test_power, snr)
            EMG_test_scale = self.scale(EEG_test_power, EMG_test_power, snr)
            EOG_EMG_test_scale = self.scale(EEG_test_power, EOG_EMG_test_power, snr)

            noisy_EOG_test = EEG_test + EOG_test * EOG_test_scale
            noisy_EMG_test = EEG_test + EMG_test * EMG_test_scale
            noisy_EOGEMG_test = EEG_test + (EOG_test + EMG_test) * EOG_EMG_test_scale

            np.save(
                f"{self.dataset_path}/noisy_EOG_snr_{snr}_train.npy", noisy_EOG_train
            )
            np.save(
                f"{self.dataset_path}/noisy_EMG_snr_{snr}_train.npy", noisy_EMG_train
            )
            np.save(
                f"{self.dataset_path}/noisy_EOGEMG_snr_{snr}_train.npy",
                noisy_EOGEMG_train,
            )
            np.save(f"{self.dataset_path}/noisy_EOG_snr_{snr}_test.npy", noisy_EOG_test)
            np.save(f"{self.dataset_path}/noisy_EMG_snr_{snr}_test.npy", noisy_EMG_test)
            np.save(
                f"{self.dataset_path}/noisy_EOGEMG_snr_{snr}_test.npy",
                noisy_EOGEMG_test,
            )

    # SNR 缩放
    @staticmethod
    def scale(p_c, p_n, snr):
        scale = np.sqrt(p_c / (p_n * (10 ** (snr / 10.0)))) if p_n > 0 else 0
        return scale


if __name__ == "__main__":
    processor = ECGDataProcessor()
    processor.preprocess()
