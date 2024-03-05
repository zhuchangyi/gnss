import pandas as pd
import os
import gnss_lib.coordinates as coord
import numpy as np

og_path = os.path.join("../kaggle2023")
data_path = os.path.join(og_path, r"sdc2023\train")
filtered_path = os.path.join(og_path, "filtered_data")
print(os.path.exists(filtered_path))


def wlsmove():
    for trace in os.listdir(filtered_path):
        phone_path = os.path.join(filtered_path, trace)
        for phones in os.listdir(phone_path):
            gnss_data = pd.read_csv(os.path.join(phone_path, phones, "gnss_data.csv"))
            wls_XYZ = np.array(
                [gnss_data['WlsPositionXEcefMeters'], gnss_data['WlsPositionYEcefMeters'],
                 gnss_data['WlsPositionZEcefMeters']])
            utc1 = np.array([gnss_data['utcTimeMillis'].unique()])
            wls_XYZ = np.transpose(wls_XYZ)
            wls = coord.LocalCoord.from_ecef(wls_XYZ[0])
            wls_ned = wls.ecef2ned(wls_XYZ)
            wls_ned_df = pd.DataFrame(wls_ned, columns=['wls_ned_X', 'wls_ned_Y', 'wls_ned_Z'])

            # 将更新后的DataFrame保存到CSV文件中
            gnss_data = pd.concat([gnss_data, wls_ned_df], axis=1)
            # print(gnss_data.head())
            # gnss_data.to_csv(os.path.join(phone_path, phones, "gnss_data.csv"), index=False)
            # print(wls_ned_df.head())
            # print(np.array(wls_ned))
            gt = pd.read_csv(os.path.join(phone_path, phones, "ground_truth.csv"))
            utc2 = np.array([gt['UnixTimeMillis']])
            if (utc1.shape[1] - utc2.shape[1] != 0):
                raise ValueError("gt and obs DIid not match ")
            gt_geo = gt[['LatitudeDegrees', 'LongitudeDegrees', 'AltitudeMeters']].to_numpy()
            gt_ecef = coord.geodetic2ecef(gt_geo)
            gt_ned = wls.ecef2ned(gt_ecef)
            print(gt_ned.shape, wls_ned.shape)
            # true_correction = gt_ned - wls_ned
            gt_ned_df = pd.DataFrame(gt_ned, columns=['gt_ned_X', 'gt_ned_Y', 'gt_ned_X'])
            # true_correction_df = pd.DataFrame(true_correction,columns=['ture_X','ture_Y','ture_Z'])
            # gt = pd.concat([gt,gt_ned_df,true_correction_df], axis=1)
            # print(gt.head())
            # gt.to_csv(os.path.join(phone_path,phones,"ground_truth.csv"), index=False)


def process_gnss_data(phone_path):
    gnss_data = pd.read_csv(os.path.join(phone_path, "gnss_data.csv"))

    # 转换WLS位置到NED坐标
    wls_XYZ = np.array([gnss_data['WlsPositionXEcefMeters'],
                        gnss_data['WlsPositionYEcefMeters'],
                        gnss_data['WlsPositionZEcefMeters']]).T

    # 只取唯一的WLS位置，对应于每个时间戳
    unique_wls_XYZ = wls_XYZ[~gnss_data['utcTimeMillis'].duplicated(), :]
    unique_utc = gnss_data['utcTimeMillis'].unique()

    # 坐标转换
    wls = coord.LocalCoord.from_ecef(unique_wls_XYZ[0])
    wls_ned = wls.ecef2ned(unique_wls_XYZ)
    wls_ned_df = pd.DataFrame(wls_ned, columns=['wls_ned_X', 'wls_ned_Y', 'wls_ned_Z'])

    # 创建带有时间戳的DataFrame
    wls_ned_df['utcTimeMillis'] = unique_utc

    # 将转换后的坐标添加到原始gnss数据中
    merged_gnss_data = pd.merge(gnss_data, wls_ned_df, on='utcTimeMillis', how='left')

    # 保存结果
    merged_gnss_data.to_csv(os.path.join(phone_path, "gnss_data.csv"), index=False)


def process_all_gnss_data(filtered_path):
    for trace in os.listdir(filtered_path):
        phone_path = os.path.join(filtered_path, trace)
        for phones in os.listdir(phone_path):
            process_gnss_data(os.path.join(phone_path, phones))


# 使用函数处理所有数据
process_all_gnss_data(filtered_path)

def process_ground_truth(phone_path):
    # Read the ground_truth and processed gnss data
    ground_truth = pd.read_csv(os.path.join(phone_path, "ground_truth.csv"))
    gnss_data_with_ned = pd.read_csv(os.path.join(phone_path, "gnss_data.csv"))
    ground_truth.rename(columns={'UnixTimeMillis': 'utcTimeMillis'}, inplace=True)
    ned_columns = ['utcTimeMillis', 'wls_ned_X', 'wls_ned_Y', 'wls_ned_Z']

    gnss_ned_data = gnss_data_with_ned[ned_columns]
    gnss_ned_data_unique = gnss_ned_data.drop_duplicates(subset=['utcTimeMillis'])

    merged_ground_truth = pd.merge(ground_truth, gnss_ned_data_unique, on='utcTimeMillis', how='left')

    merged_ground_truth.to_csv(os.path.join(phone_path, "ground_truth.csv"), index=False)


def process_all_ground_truth(filtered_path):
    for trace in os.listdir(filtered_path):
        phone_path = os.path.join(filtered_path, trace)
        for phones in os.listdir(phone_path):
            process_ground_truth(os.path.join(phone_path, phones))


# 使用函数处理所有数据
process_all_ground_truth(filtered_path)

def process_gt_ned(phone_path):
    ground_truth = pd.read_csv(os.path.join(phone_path, "ground_truth.csv"))
    wls_XYZ = np.array(
        [ground_truth['WlsPositionXEcefMeters'], ground_truth['WlsPositionYEcefMeters'],
         ground_truth['WlsPositionZEcefMeters']]).T
    wls_ned = np.array([ground_truth['wls_ned_X'], ground_truth['wls_ned_Y'],
                        ground_truth['wls_ned_Z']]).T
    wls = coord.LocalCoord.from_ecef(wls_XYZ[0])
    gt_geo = ground_truth[['LatitudeDegrees', 'LongitudeDegrees', 'AltitudeMeters']].to_numpy()
    gt_ecef = coord.geodetic2ecef(gt_geo)
    gt_ned = wls.ecef2ned(gt_ecef)
    true_correction = gt_ned - wls_ned
    gt_ned_df = pd.DataFrame(gt_ned, columns=['gt_ned_X', 'gt_ned_Y', 'gt_ned_Z'])
    ture_correct = pd.DataFrame(true_correction, columns=['ture_correct_X', 'ture_correct_Y', 'ture_correct_Z'])
    gt = pd.concat([ground_truth, gt_ned_df, ture_correct], axis=1)
    gt.to_csv(os.path.join(phone_path, "ground_truth.csv"), index=False)

def process_all_ground(filtered_path):
    for trace in os.listdir(filtered_path):
        phone_path = os.path.join(filtered_path, trace)
        for phones in os.listdir(phone_path):
             #delete_obs(os.path.join(phone_path, phones))
            process_gt_ned(os.path.join(phone_path, phones))





def delete_gt(path):
    ground_truth = pd.read_csv(os.path.join(path, "ground_truth.csv"))
    ground_truth = ground_truth.iloc[:, :13]
    ground_truth.to_csv(os.path.join(path, "ground_truth.csv"), index=False)

def delete_obs(path):
    ground_truth = pd.read_csv(os.path.join(path, "gnss_data.csv"))
    ground_truth = ground_truth.iloc[:, :-6]
    ground_truth.to_csv(os.path.join(path, "gnss_data.csv"), index=False)

process_all_ground(filtered_path)