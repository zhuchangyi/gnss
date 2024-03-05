import pandas as pd
import logging
import os

# Set up logging
# logging.basicConfig(filename='timestamp_count.log', level=logging.INFO,
#                     format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
#
# for subdir, dir, files in os.walk(r'../kaggle2023/filtered_data'):
#     gnss_file = os.path.join(subdir, 'gnss_data.csv')
#     if os.path.exists(gnss_file):
#         gnss_df = pd.read_csv(gnss_file)
#         unique_timestamps = gnss_df['utcTimeMillis'].unique()
#         logging.info(f'The number of unique timestamps is: {len(unique_timestamps)}')

with open('timestamp_count.log', 'r') as f:
    lines = f.readlines()
    num_timestamps = 0
    for line in lines:
        num_timestamps += int(line.split(' ')[-1])
    print(f'The total number of unique timestamps is: {num_timestamps}')
    # The total number of unique timestamps is: 251671


