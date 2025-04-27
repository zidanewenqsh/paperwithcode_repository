from hjimi_tools import FileCollector
if __name__ == '__main__':
    # paths = ["path/to/project1", "path/to/project2"]
    # paths = [r"D:\SXProjects\Depth_ForeGround_TestEXE_Compress"]
    # extensions = ['.h']
    # output_path = 'foreground-h.md'
    paths = [r"D:\SXProjects\depth_foreground_testexe_compress"]
    extensions = ['.h', '.hpp', '.cpp', '.cc', '.cu', '.c']
    output_path = 'foreground.md'

    collector = FileCollector(paths, extensions, output_path)
    collector.run()