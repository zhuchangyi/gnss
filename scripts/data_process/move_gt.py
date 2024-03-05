from pathlib import Path
import shutil

# 获取当前脚本的绝对路径
current_script_path = Path(__file__).resolve()
# 获取根目录路径
root_path = current_script_path.parents[2]
# 构建到 'data/processed' 和 'data/raw' 的路径
filtered_path = root_path / "data" / "processed"
data_raw_path = root_path / "data" / "raw" / "sdc2023" / "train"

# 遍历data_raw_path目录下的所有子目录和文件
for path in data_raw_path.rglob('ground_truth.csv'):
    # 目标目录路径
    processed_trace_path = filtered_path / path.relative_to(data_raw_path)
    processed_trace_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在
    shutil.copy(path, processed_trace_path)  # 复制文件
