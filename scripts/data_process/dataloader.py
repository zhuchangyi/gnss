import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class GNSSDataset(Dataset):
    def __init__(self, root_dir, num_satellites=20):
        self.root_dir = root_dir
        self.num_satellites = num_satellites
        self.data, self.labels = self._load_data()

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
                         'SvClockBiasMeters',
                         'WlsPositionXEcefMeters',
                         'WlsPositionYEcefMeters',
                         'WlsPositionZEcefMeters',
                         ]].fillna(0).to_numpy()


                    # Padding if less than num_satellites
                    if len(timestamp_data) < self.num_satellites:
                        padding = np.zeros((self.num_satellites - len(timestamp_data), gnss_features.shape[1]))
                        gnss_features = np.vstack([gnss_features, padding])

                    ground_truth = timestamp_data[['ture_correct_X', 'ture_correct_Y', 'ture_correct_Z']].iloc[
                        0].fillna(0).to_numpy()

                    all_features.append(gnss_features)
                    all_labels.append(ground_truth)
                    # print(len(all_features),len(all_features[0]))
                    # print(len(all_labels), len(all_labels[0]))




        return all_features, all_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]).float(), torch.tensor(self.labels[idx]).float()


# Usage
root_dir = './data/processed'
dataset = GNSSDataset(root_dir)
dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

# Iterate through the dataloader
for gnss_batch, ground_truth_batch in dataloader:
    # Process your batch

    pass