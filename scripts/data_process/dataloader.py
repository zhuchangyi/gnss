import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class GNSSDataset(Dataset):
    def __init__(self, root_dir, num_satellites=60):
        self.root_dir = Path(root_dir)
        self.num_satellites = num_satellites
        self.data, self.labels = self._load_data()

    def _correction_to_one_hot(self, correction):
        error = np.clip(np.round(correction), -10, 10)
        one_hot_vector = np.zeros(21)
        index = int(error + 10)
        one_hot_vector[index] = 1
        return one_hot_vector

    def _calculate_distance(self, x1, y1, z1, x2, y2, z2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

    def _load_data(self):
        all_features = []
        all_labels = []
        for subdir, dirs, files in os.walk(self.root_dir):
            gnss_file = os.path.join(subdir, 'gnss_data.csv')
            ground_truth_file = os.path.join(subdir, 'ground_truth.csv')

            if os.path.exists(gnss_file) and os.path.exists(ground_truth_file):
                gnss_df = pd.read_csv(gnss_file)
                ground_truth_df = pd.read_csv(ground_truth_file)

                # Merge on utcTimeMillis to align data by time
                columns_to_remove = ['WlsPositionXEcefMeters', 'WlsPositionYEcefMeters', 'WlsPositionZEcefMeters']
                gnss_df = gnss_df.drop(columns=columns_to_remove)
                merged_df = pd.merge(gnss_df, ground_truth_df, on='utcTimeMillis')
                unique_timestamps = merged_df['utcTimeMillis'].unique()

                for timestamp in unique_timestamps:
                    timestamp_data = merged_df[merged_df['utcTimeMillis'] == timestamp]

                    # Calculate pseudorange residuals and LOS vectors for each satellite
                    pseudorange_residuals = []
                    los_vectors = []
                    for index, row in timestamp_data.iterrows():
                        distance = self._calculate_distance(
                            row['WlsPositionXEcefMeters'], row['WlsPositionYEcefMeters'], row['WlsPositionZEcefMeters'],
                            row['SvPositionXEcefMeters'], row['SvPositionYEcefMeters'], row['SvPositionZEcefMeters']
                        )

                        residual = row['RawPseudorangeMeters'] - distance
                        pseudorange_residuals.append(residual)

                        los_vector = np.array([
                            row['SvPositionXEcefMeters'] - row['WlsPositionXEcefMeters'],
                            row['SvPositionYEcefMeters'] - row['WlsPositionYEcefMeters'],
                            row['SvPositionZEcefMeters'] - row['WlsPositionZEcefMeters']
                        ])
                        # Normalize the los_vector
                        los_vector_normalized = los_vector / np.linalg.norm(los_vector)
                        los_vectors.append(los_vector_normalized)

                    # Convert lists to numpy arrays
                    pseudorange_residuals = np.array(pseudorange_residuals)
                    los_vectors = np.stack(los_vectors)

                    # Prepare GNSS features, exclude SvPositionXEcefMeters, SvPositionYEcefMeters, SvPositionZEcefMeters
                    gnss_features = timestamp_data[[
                        'Cn0DbHz', 'IonosphericDelayMeters', 'TroposphericDelayMeters',
                        'SvElevationDegrees', 'SvAzimuthDegrees',
                        'PseudorangeRateMetersPerSecond', 'RawPseudorangeMeters',
                        'PseudorangeRateUncertaintyMetersPerSecond',
                        'SvVelocityXEcefMetersPerSecond', 'SvVelocityYEcefMetersPerSecond',
                        'SvVelocityZEcefMetersPerSecond',
                        'SvClockBiasMeters'
                    ]].fillna(0).to_numpy()

                    # Add LOS vectors and pseudorange residuals to features
                    gnss_features = np.hstack([gnss_features, los_vectors, pseudorange_residuals[:, np.newaxis]])

                    # Pad features if necessary
                    if gnss_features.shape[0] < self.num_satellites:
                        padding = np.zeros((self.num_satellites - gnss_features.shape[0], gnss_features.shape[1]))
                        gnss_features = np.vstack([gnss_features, padding])

                    # Convert corrections to one-hot vectors
                    correction_x = self._correction_to_one_hot(timestamp_data['true_correction_x'].iloc[0])
                    correction_y = self._correction_to_one_hot(timestamp_data['true_correction_y'].iloc[0])
                    correction_z = self._correction_to_one_hot(timestamp_data['true_correction_z'].iloc[0])

                    all_features.append(gnss_features)
                    all_labels.append(np.stack([correction_x, correction_y, correction_z], axis=-1))

        return np.array(all_features), np.array(all_labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float), torch.tensor(self.labels[idx], dtype=torch.float)

# Usage example
root_dir = '/Users/park/PycharmProjects/gnss/data/processed_data/2020-06-25-00-34-us-ca-mtv-sb-101'  # Adjust as needed


dataset = GNSSDataset(root_dir)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for features, labels in dataloader:
    print(features)
    print(labels)
    print(f"Batch features shape: {features.shape}")  # 打印特征的形状
    print(f"Batch labels shape: {labels.shape}")      # 打印标签的形状
    break
