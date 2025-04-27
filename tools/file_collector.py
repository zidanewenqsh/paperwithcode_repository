from pathlib import Path

class FileCollector:
    """
    A class to collect files with specified extensions from given paths
    and write their contents into a Markdown file.
    """

    # File extension to language mapping
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
        Initialize the FileCollector class instance.

        Parameters:
        paths: List of file or directory paths to collect files from.
        extensions: List of file extensions to collect.
        output_path: Path to the output markdown file.
        """
        self.paths = [Path(path) for path in paths]
        self.extensions = extensions
        self.output_path = Path(output_path)
        self.path_to_files = {}

    def collect_files(self, path):
        """
        Recursively collect files that match the given extensions from the specified path.

        Parameters:
        path: Path object representing the directory to be traversed.

        Returns:
        A list of file paths matching the extensions.
        """
        files_to_save = []
        for p in path.iterdir():
            if p.is_dir():
                # Recursively collect files from subdirectories
                files_to_save.extend(self.collect_files(p))
            elif p.suffix in self.extensions:
                files_to_save.append(p)
        return files_to_save

    def collect_all_files(self):
        """
        Traverse all the provided paths and collect files matching the extensions.
        """
        for path in self.paths:
            if path.exists():
                if path.is_dir():
                    # Recursively collect files from directories
                    self.path_to_files[path] = self.collect_files(path)
                elif path.suffix in self.extensions:
                    self.path_to_files[path] = [path]

    def write_to_markdown(self):
        """
        Write the collected files and their contents into a Markdown file.
        """
        # Ensure the output file's parent directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the files content into the markdown output
        with self.output_path.open('w', encoding='utf-8') as f:
            processed_dirs = set()  # Track which root directories have been processed

            for base_path, files in self.path_to_files.items():
                root_project_name = base_path.name

                # Write the project root name as the top-level heading
                if root_project_name not in processed_dirs:
                    f.write(f"# {root_project_name}\n\n")
                    processed_dirs.add(root_project_name)

                # Process each file within the root directory
                for file_path in files:
                    relative_path = file_path.relative_to(base_path)
                    language = self.LANGUAGE_MAP.get(file_path.suffix, 'Plain Text')  # Default to 'Plain Text' if not found

                    # Write the file's name as a secondary heading, along with its relative path
                    f.write(f"## {file_path.name}\n")
                    f.write(f"File Path: `{relative_path}`\n")
                    f.write(f"```{language}\n")  # Language-specific code block

                    try:
                        # Read and write the file's content to the markdown file
                        with file_path.open('r', encoding='utf-8') as file:
                            f.write(file.read())
                        f.write("\n```\n\n")  # End the code block and add a newline
                    except UnicodeDecodeError:
                        f.write("**Error decoding file.**\n\n")  # Handle decoding errors

    def run(self):
        """
        Run the file collection and markdown writing process.
        """
        self.collect_all_files()
        self.write_to_markdown()

if __name__ == '__main__':
    # Example usage
    paths = [r"D:\MyProjects\graph_demo1", r"D:\MyProjects\my_api_03"]
    extensions = ['.py', '.js', '.md', '.txt']
    output_path = 'output.md'

    collector = FileCollector(paths, extensions, output_path)
    collector.run()
