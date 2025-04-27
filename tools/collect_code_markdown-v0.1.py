from pathlib import Path

def collect_and_write_files(paths, extensions, output_path):
    """
    遍历给定的路径列表，找到所有扩展名在extensions中的文件，并将它们的内容保存到指定的输出文件中，输出为Markdown格式。
    
    参数:
    paths: 文件或文件夹的路径列表。
    extensions: 接受的文件后缀名列表。
    output_path: 输出文件的路径。
    """
    
    # 文件后缀名与语言的映射
    language_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.java': 'java',
        '.html': 'html',
        '.css': 'css',
        '.cpp': 'cpp',
        '.h': 'cpp',
        '.ts': 'typescript',
        '.rb': 'ruby',
        '.go': 'go',
        '.sh': 'bash',
        '.md': 'markdown',
        '.ipynb': 'jupyter',  # Jupyter Notebooks
    }

    def collect_files(path, extensions):
        files_to_save = []
        # 递归收集符合条件的文件
        for p in path.iterdir():
            if p.is_dir():
                # 递归调用 collect_files 收集子目录中的文件
                files_to_save.extend(collect_files(p, extensions))  # 合并子目录文件
            elif p.suffix in extensions:
                files_to_save.append(p)
        return files_to_save

    # 存储所有路径到文件的映射
    path_to_files = {}

    # 遍历路径，收集所有符合条件的文件
    for path_string in paths:
        path = Path(path_string)
        if path.exists():
            if path.is_dir():
                # 递归收集文件
                path_to_files[path] = collect_files(path, extensions)
            elif path.suffix in extensions:
                path_to_files[path] = [path]

    # 写入文件内容到Markdown文件
    output_file_path = Path(output_path)

    # 确保输出文件的目录存在
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    # 将文件内容写入输出文件，以Markdown格式
    with output_file_path.open('w', encoding='utf-8') as f:
        # 记录每个根目录
        processed_dirs = set()  # 用于记录已经处理过的根目录

        # 遍历字典，获取每个根目录及其文件
        for base_path, files in path_to_files.items():
            # 获取根工程名（即路径的最后一部分）
            root_project_name = base_path.name

            # 检查是否已经写入过这个根目录的一级标题
            if root_project_name not in processed_dirs:
                f.write(f"# {root_project_name}\n\n")
                processed_dirs.add(root_project_name)

            for file_path in files:
                # 计算相对路径
                relative_path = file_path.relative_to(base_path)
                
                # 获取文件的语言标识
                language = language_map.get(file_path.suffix, 'text')  # 默认使用 'text' 作为语言标识
                
                # 写入二级标题，文件相对路径引用
                f.write(f"## {file_path.name}\n")
                f.write(f"文件路径: `{relative_path}`\n")
                f.write(f"```{language}\n")  # 文件扩展名对应的语言标识

                try:
                    # 写入文件内容
                    with file_path.open('r', encoding='utf-8') as file:
                        f.write(file.read())
                    
                    f.write("\n```\n\n")  # 代码框结束并换行
                except UnicodeDecodeError as e:
                    print(f"Error decoding file {file_path}")



# 使用示例
paths = [r"D:\MyProjects\graph_demo1", r"D:\MyProjects\my_api_03"]
# paths = [r"D:\MyProjects\my_api_03"]
extensions = ['.py', '.ipynb', '.md', '.js', '.html', '.css', '.java', '.cpp', '.h', '.ts', '.rb', '.go', '.sh']  # 支持更多语言
output_path = 'output.md'

collect_and_write_files(paths, extensions, output_path)
