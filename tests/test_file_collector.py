# tests/test_file_collector.py

import unittest
from pathlib import Path
from tools.file_collector import FileCollector

class TestFileCollector(unittest.TestCase):
    def setUp(self):
        """
        在每个测试方法运行之前调用，用于设置测试环境。
        创建一个临时的测试目录和一些测试文件，以便测试 FileCollector 类的功能。
        """
        # 定义测试数据的目录路径
        self.test_dir = Path("tests/test_data")
        
        # 创建测试数据目录，包括所有必要的父目录
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # 在测试目录中创建一个 .txt 文件，内容为 "Hello, World!"
        (self.test_dir / "test1.txt").write_text("Hello, World!", encoding='utf-8')
        
        # 在测试目录中创建一个 .md 文件，内容为 "# Markdown File"
        (self.test_dir / "test2.md").write_text("# Markdown File", encoding='utf-8')

        # 定义输出文件的路径，将在测试中生成
        self.output_file = Path("tests/output.md")

    def tearDown(self):
        """
        在每个测试方法运行之后调用，用于清理测试环境。
        删除之前创建的测试文件和目录，以确保每个测试都是独立的，不受其他测试的影响。
        """
        # 检查输出文件是否存在，如果存在则删除
        if self.output_file.exists():
            self.output_file.unlink()
        
        # 遍历测试数据目录中的所有文件，并删除它们
        for file in self.test_dir.iterdir():
            file.unlink()
        
        # 删除测试数据目录本身
        self.test_dir.rmdir()

    def test_collect_and_write_files(self):
        """
        测试 FileCollector 类的主要功能：
        1. 收集指定目录下符合扩展名的文件。
        2. 将这些文件的内容写入到一个 Markdown 格式的输出文件中。
        """
        # 设置测试用的路径列表，包含测试数据目录的路径
        paths = [str(self.test_dir)]
        
        # 设置接受的文件扩展名列表，这里只接受 '.txt' 文件
        extensions = ['.txt']
        
        # 创建 FileCollector 类的实例，传入路径、扩展名和输出文件路径
        collector = FileCollector(paths, extensions, str(self.output_file))
        
        # 运行 FileCollector 实例的方法，执行文件收集和写入操作
        collector.run()

        # 断言输出文件是否被成功创建
        self.assertTrue(self.output_file.exists(), "输出文件未被创建。")
        
        # 读取输出文件的内容
        content = self.output_file.read_text(encoding='utf-8')
        
        # 断言输出文件中包含 'test1.txt' 的文件名
        self.assertIn("test1.txt", content, "输出文件中未包含预期的文件名 'test1.txt'。")
        
        # 断言输出文件中包含 'test1.txt' 文件的内容
        self.assertIn("Hello, World!", content, "输出文件中未包含预期的文件内容。")
        
        # 断言输出文件中不包含 'test2.md' 的文件名，因为 '.md' 不在扩展名列表中
        self.assertNotIn("test2.md", content, "输出文件中不应包含文件名 'test2.md'，因为其扩展名不在列表中。")

if __name__ == '__main__':
    unittest.main()
