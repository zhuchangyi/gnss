#用于gnss数据的统计分析
#数据基于原始文件包括gt加上wls的ecef数据
#缺失的数据
import os
import pandas as pd
import numpy as np
import logging
import shutil
from pathlib import Path

current_script_path = Path(__file__).resolve()
root_path = current_script_path.parents[2]
print(current_script_path,root_path)

rawdata_path = root_path / "data" / "raw" / "sdc2023" / "train"
processed_path = root_path / "data" / "processed_data"
log_path = root_path / "logs"
#log_file_path = log_path / 'num.log'
#logging.basicConfig(filename=log_file_path, level=logging.INFO,
                     #format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
#先是每个历元的观测数据有多少
def log_utc():
    for subdir, dirs, files in os.walk(processed_path):
        gnss_file = os.path.join(subdir, 'gnss_data.csv')
        ground_truth_file = os.path.join(subdir, 'ground_truth.csv')

        if os.path.exists(gnss_file) and os.path.exists(ground_truth_file):
            #print(subdir)
            gnss_df = pd.read_csv(gnss_file)
            gt_df = pd.read_csv(ground_truth_file)
            unique_timestamps = gnss_df['utcTimeMillis'].unique()
            gt_unique_timestamps = gt_df['utcTimeMillis'].unique()
            # 检查两个时间戳数组的长度是否相同
            if len(unique_timestamps) != len(gt_unique_timestamps):
                logging.info(
                    f"{subdir} Number of timestamps does not match between gnss ({len(unique_timestamps)}) and ground truth ({len(gt_unique_timestamps)})")
            else:
                # 检查两个时间戳数组的内容是否完全一样
                if np.array_equal(unique_timestamps.sort(), gt_unique_timestamps.sort()):
                    logging.info(f'{subdir} The number of equal timestamps in file is: {len(unique_timestamps)}')
                else:
                    logging.info(
                        f"{subdir} Timestamps in gnss and ground truth have the same length but do not match exactly.")

#log_utc()

def remove_utc():
    for trace in os.listdir(rawdata_path):
        trace_path = os.path.join(rawdata_path, trace)
        if os.path.isdir(trace_path):  # 确保trace是一个目录
            for phones in os.listdir(trace_path):
                phone_path = os.path.join(trace_path, phones)
                if os.path.isdir(phone_path):  # 确保phones是一个目录
                    out_path = os.path.join(processed_path, trace, phones)
                    gnss_csv_path = os.path.join(phone_path, "device_gnss.csv")
                    gt_path = os.path.join(phone_path, 'ground_truth.csv')

                    #处理没有匹配的时间戳
                    if os.path.exists(gnss_csv_path) and os.path.exists(gt_path):
                        gnss_df = pd.read_csv(gnss_csv_path)
                        gt_df = pd.read_csv(gt_path)
                        # 从ground_truth获取有效的utcTimeMillis
                        valid_utcs = gt_df['utcTimeMillis'].unique()
                        # 筛选gnss_df中有效的时间戳行
                        gnss_df_filtered = gnss_df[gnss_df['utcTimeMillis'].isin(valid_utcs)]
                        out_csv_path = os.path.abspath(os.path.join(out_path, "gnss_data.csv"))
                        # 获取CSV文件的父目录路径
                        out_dir = os.path.dirname(out_csv_path)
                        # 如果父目录不存在，则创建它
                        if not os.path.exists(out_dir):
                            os.makedirs(out_dir)
                        gnss_df_filtered.to_csv(out_csv_path, index=False)
                        gt_out_path = os.path.join(out_dir, 'ground_truth.csv')
                        shutil.copyfile(gt_path, gt_out_path)




#remove_utc()
#检查每个utc星座的个数
log_file_path = log_path / 'count.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)

def check_satellite_constellations():
    for subdir, dirs, files in os.walk(processed_path):
        gnss_file = os.path.join(subdir, 'gnss_data.csv')

        if os.path.exists(gnss_file):
            gnss_df = pd.read_csv(gnss_file)

            # 获取每个utcTimeMillis的记录数
            utc_counts = gnss_df['utcTimeMillis'].value_counts()

            # 对每个utcTimeMillis，获取不同星座（ConstellationType）的观测数据个数
            for utc, count in utc_counts.items():
                constellations_count = gnss_df[gnss_df['utcTimeMillis'] == utc]['ConstellationType'].value_counts()
                logging.info(f"UTC: {utc}, Total Observations: {count}")
                logging.info("Constellations and counts:")
                logging.info(constellations_count)

# 假设 processed_path 已经被正确设置
check_satellite_constellations()

