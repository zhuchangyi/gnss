import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class GNSSDataset(Dataset):
    def __init__(self, root_dir, num_satellites=20):
        self.root_dir = root_dir
        self.num_satellites = num_satellites
        self.data, self.labels = self._load_data()

    def _correction_to_one_hot(self,correction):
        error = np.clip(np.round(correction), -10, 10)
        one_hot_vector = np.zeros(21)
        index = int(error + 10)
        one_hot_vector[index] = 1
        return one_hot_vector

    def _load_data(self):
        all_features = []
        all_labels = []
        for subdir, dirs, files in os.walk(self.root_dir):
            gnss_file = os.path.join(subdir, 'gnss_data.csv')
            ground_truth_file = os.path.join(subdir, 'ground_truth.csv')

            if os.path.exists(gnss_file) and os.path.exists(ground_truth_file):
                gnss_df = pd.read_csv(gnss_file)
                ground_truth_df = pd.read_csv(ground_truth_file)

                merged_df = pd.merge(gnss_df, ground_truth_df, on='utcTimeMillis')
                unique_timestamps = merged_df['utcTimeMillis'].unique()

                for timestamp in unique_timestamps:
                    timestamp_data = merged_df[merged_df['utcTimeMillis'] == timestamp]

                    gnss_features = timestamp_data[
                        ['PseudorangeRateMetersPerSecond', 'PseudorangeRateUncertaintyMetersPerSecond',
                         'RawPseudorangeMeters', 'RawPseudorangeUncertaintyMeters', 'AccumulatedDeltaRangeMeters',
                         'SvPositionXEcefMeters', 'SvPositionYEcefMeters', 'SvPositionZEcefMeters',
                         'SvVelocityXEcefMetersPerSecond', 'SvVelocityYEcefMetersPerSecond',
                         'SvVelocityZEcefMetersPerSecond',
                         'SvClockBiasMeters'
                        ]].fillna(0).to_numpy()

                    if len(timestamp_data) < self.num_satellites:
                        padding = np.zeros((self.num_satellites - len(timestamp_data), gnss_features.shape[1]))
                        gnss_features = np.vstack([gnss_features, padding])

                    correction_x = self._correction_to_one_hot(timestamp_data['ture_correct_X'].iloc[0])
                    correction_y = self._correction_to_one_hot(timestamp_data['ture_correct_Y'].iloc[0])
                    correction_z = self._correction_to_one_hot(timestamp_data['ture_correct_Z'].iloc[0])

                    all_features.append(gnss_features)
                    all_labels.append(np.stack([correction_x, correction_y, correction_z], axis=1))

        return np.array(all_features), np.array(all_labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]).float(), torch.tensor(self.labels[idx]).float()


# Usage

current_script_path = Path(__file__).resolve()
root_path = current_script_path.parents[2]
# 构建到 'data/processed' 和 'data/raw' 的路径
filtered_path = root_path / "data" / "processed"
if os.path.exists(filtered_path):
    print("yes")
dataset = GNSSDataset(filtered_path)
dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

# Iterate through the dataloader
for gnss_batch, ground_truth_batch in dataloader:
    # Process your batch

    pass
