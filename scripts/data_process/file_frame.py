import os


class File:
    def __init__(self, name, is_dir):
        self.name = name
        self.is_dir = is_dir

    def __repr__(self):
        if self.is_dir:
            return f"├── {self.name}"
        else:
            return f"\t└── {self.name}"


def main():
    # 获取当前工作目录
    og_path = os.path.join(r"..\kaggle2023")
    filtered_path = os.path.join(og_path, "filtered_data")
    if os.path.exists(filtered_path):
        print(filtered_path)

    # 遍历文件夹及其子文件夹
    for root, _, files in os.walk(filtered_path):
        # 为每个文件或文件夹创建一个 File 对象
        for file in files:
            if isinstance(file, File):
                file_obj = file
            else:
                file_obj = File(file, os.path.isdir(os.path.join(root, file)))

        # 使用 print() 函数来打印 File 对象的树状表示
        print("".join(map(str, [File(root, True)] + [file_obj for file_obj in files])))


if __name__ == "__main__":
    main()
