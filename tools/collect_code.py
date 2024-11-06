from pathlib import Path
# from icecream import ic

def collect_and_write_files(paths, extensions, output_path):
    """
    遍历给定的路径列表，找到所有扩展名在extensions中的文件，并将它们的内容保存到指定的输出文件中。
    
    参数:
    paths: 文件或文件夹的路径列表。
    extensions: 接受的文件后缀名列表。
    output_path: 输出文件的路径。
    """
    script_path = Path(__file__).resolve()  # 获取当前脚本的绝对路径
    def collect_files(path):
        # 递归收集符合条件的文件
        for p in path.iterdir():
            if p.is_dir():
                collect_files(p)
            elif p.suffix in extensions and p.resolve() != script_path:
                files_to_save.append(p)
    
    files_to_save = []
    output_file_path = Path(output_path)
    print(output_file_path)
    # 确保输出文件的目录存在
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 遍历所有路径，收集文件
    for path_string in paths:
        path = Path(path_string)
        if path.exists():
            if path.is_dir():
                collect_files(path)
            elif path.suffix in extensions:
                files_to_save.append(path)
    
    # 将文件内容写入输出文件
    with output_file_path.open('w', encoding='utf-8') as f:
        for file_path in files_to_save:
            # 写入文件的绝对路径
            try:
                f.write(f"{file_path.resolve()}\n")
                #写入文件内容
                with file_path.open('r', encoding='utf-8') as file:
                    f.write(file.read())
                    f.write("\n\n")  # 在文件内容之间加入空行以区分不同的文件内容
            except UnicodeDecodeError as e:
		            # 有些文件可能不是utf-8编码的
                print(f"Error decoding file {file_path}")

# 使用示例
# paths = [r"D:\Projects\smplx"]
paths = [r"D:\Projects\smplify-x"]
extensions = ['.py', '.ipynb', '.md']
# output_path = 'smplx_code.txt'
output_path = r'.\saves\smplify-x.txt'
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
collect_and_write_files(paths, extensions, output_path)
