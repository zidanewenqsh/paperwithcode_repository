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
    
    def collect_files(path):
        # 递归收集符合条件的文件
        for p in path.iterdir():
            if p.is_dir():
                collect_files(p)
            elif p.suffix in extensions:
                files_to_save.append(p)
    
    files_to_save = []
    output_file_path = Path(output_path)
    
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
    
    # 将文件内容写入输出文件，以Markdown格式
    with output_file_path.open('w', encoding='utf-8') as f:
        for file_path in files_to_save:
            # 找到文件所在的路径，并计算相对路径
            for base_path in paths:
                base_path = Path(base_path)
                if file_path.is_relative_to(base_path):  # 判断文件是否在当前路径下
                    relative_path = file_path.relative_to(base_path)  # 计算相对路径
                    break
            # # 文件相对路径
            # relative_path = file_path.relative_to(Path(paths[0]).resolve())
            
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

