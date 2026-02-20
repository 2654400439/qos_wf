import os
import json

def rename_files(directory):
    # 获取文件夹中的所有文件名
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # 初始化一个字典来存储类别和对应的递增数字
    category_map = {}
    category_counter = 1

    for file in files:
        # 分割文件名，获取类别部分
        category = file.split('-')[0]

        # 如果该类别还没有分配数字，分配一个新的数字
        if category not in category_map:
            category_map[category] = category_counter
            category_counter += 1

        # 获取该文件的新的前缀数字
        new_prefix = category_map[category]

        # 获取原文件的后缀
        suffix = str(int(file.split('-')[-1].split('.')[0]))

        # 构造新的文件名
        new_name = f"{new_prefix}-{suffix}.txt"

        # 获取文件的完整路径
        old_path = os.path.join(directory, file)
        new_path = os.path.join(directory, new_name)

        # 重命名文件
        os.rename(old_path, new_path)

        print(f"Renamed: {old_path} to {new_path}")

        # 保存映射关系到文件
    mapping_file = os.path.join('../preprocessed_data', "RF_category_mapping.json")
    with open(mapping_file, "w") as f:
        json.dump(category_map, f, indent=4)

    print(f"Mapping saved to: {mapping_file}")
    print(f"Total categories: {category_counter - 1}")


# 调用函数并传递文件夹路径
rename_files('../preprocessed_data/RF')
