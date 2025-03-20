import os

# 定义文件夹路径
folder_path = "/share/project/lxh/datasets/tf_datasets/bridge_orig/1.0.0/"

# 定义文件名前缀和后缀
prefix = "bridge_dataset-val.tfrecord-"
suffix = "-of-00128"

# 生成所有可能的文件名并检查是否存在
missing_files = []
for i in range(128):
    # 生成5位编号的文件名
    filename = f"{prefix}{str(i).zfill(5)}{suffix}"
    full_path = os.path.join(folder_path, filename)
    if not os.path.exists(full_path):
        missing_files.append(filename)

# 打印缺失的文件
print("缺失的文件有：")
for file in missing_files:
    print(file)