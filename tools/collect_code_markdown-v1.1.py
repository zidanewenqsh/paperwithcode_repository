# 封装成类
# example_package/file_collector.py

from pathlib import Path

class FileCollector:
    """
    A class to collect files with specified extensions from given paths
    and write their contents into a Markdown file.
    """

    # 文件后缀名与语言的映射
    LANGUAGE_MAP = {
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

    def __init__(self, paths, extensions, output_path):
        """
        初始化 FileCollector 类实例。

        参数:
        paths: 文件或文件夹的路径列表。
        extensions: 接受的文件后缀名列表。
        output_path: 输出文件的路径。
        """
        self.paths = [Path(path) for path in paths]
        self.extensions = extensions
        self.output_path = Path(output_path)
        self.path_to_files = {}

    def collect_files(self, path):
        """
        递归收集指定路径下符合条件的文件。

        参数:
        path: Path 对象，表示要遍历的路径。

        返回:
        符合条件的文件路径列表。
        """
        files_to_save = []
        for p in path.iterdir():
            if p.is_dir():
                # 递归调用 collect_files 收集子目录中的文件
                files_to_save.extend(self.collect_files(p))
            elif p.suffix in self.extensions:
                files_to_save.append(p)
        return files_to_save

    def collect_all_files(self):
        """
        遍历所有给定的路径，收集符合条件的文件。
        """
        for path in self.paths:
            if path.exists():
                if path.is_dir():
                    # 递归收集文件
                    self.path_to_files[path] = self.collect_files(path)
                elif path.suffix in self.extensions:
                    self.path_to_files[path] = [path]

    def write_to_markdown(self):
        """
        将收集到的文件内容写入 Markdown 文件。
        """
        # 确保输出文件的目录存在
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # 将文件内容写入输出文件，以Markdown格式
        with self.output_path.open('w', encoding='utf-8') as f:
            processed_dirs = set()  # 用于记录已经处理过的根目录

            for base_path, files in self.path_to_files.items():
                root_project_name = base_path.name

                # 检查是否已经写入过这个根目录的一级标题
                if root_project_name not in processed_dirs:
                    f.write(f"# {root_project_name}\n\n")
                    processed_dirs.add(root_project_name)

                for file_path in files:
                    relative_path = file_path.relative_to(base_path)
                    language = self.LANGUAGE_MAP.get(file_path.suffix, 'text')  # 默认使用 'text' 作为语言标识

                    # 写入二级标题，文件相对路径引用
                    f.write(f"## {file_path.name}\n")
                    f.write(f"文件路径: `{relative_path}`\n")
                    f.write(f"```{language}\n")  # 文件扩展名对应的语言标识

                    try:
                        with file_path.open('r', encoding='utf-8') as file:
                            f.write(file.read())
                        f.write("\n```\n\n")  # 代码框结束并换行
                    except UnicodeDecodeError:
                        f.write("**Error decoding file.**\n\n")

    def run(self):
        """
        运行文件收集和写入流程。
        """
        self.collect_all_files()
        self.write_to_markdown()

# paths = ["path/to/project1", "path/to/project2"]
paths = [r"D:\MyProjects\graph_demo1", r"D:\MyProjects\my_api_03"]
extensions = ['.py', '.js', '.md']
output_path = 'output1.md'

collector = FileCollector(paths, extensions, output_path)
collector.run()