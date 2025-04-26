import os

def analyze_python_files(directory):
    short_files = []
    keyword_files = []
    keyword = "your_dataset_folder_name"

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        non_empty_lines = [line for line in lines if line.strip()]
                        if len(non_empty_lines) < 2:
                            short_files.append(file_path)
                        if any(keyword in line for line in lines):
                            keyword_files.append(file_path)
                except Exception as e:
                    print(f"无法读取文件 {file_path}，原因：{e}")
    return short_files, keyword_files

if __name__ == "__main__":
    directory_to_check = input("请输入要检测的目录路径：")
    short_files, keyword_files = analyze_python_files(directory_to_check)

    if short_files:
        print("\n以下 Python 文件的有效行数少于 2 行：")
        for file in short_files:
            print(file)
    else:
        print("\n没有找到有效行数少于 2 行的 Python 文件。")

    if keyword_files:
        print("\n以下 Python 文件中包含 'your_dataset_folder_name'：")
        for file in keyword_files:
            print(file)
    else:
        print("\n没有找到包含 'your_dataset_folder_name' 的 Python 文件。")