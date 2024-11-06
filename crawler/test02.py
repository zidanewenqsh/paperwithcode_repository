from pathlib import Path
from functools import partial
def custom_print(*messages, file_path=None, mode='a'):
    """
    打印多个消息到控制台，并可选地将消息写入文件。

    参数:
    - messages (str): 要打印的消息，可以是多个。
    - file_path (str): 要写入的文件路径，默认为 None。如果提供了文件路径，则将消息写入该文件。
    - mode (str): 文件写入模式，默认是 'a'（追加模式）。可以为 'w'（覆盖模式）或 'a'（追加模式）。
    """
    # 打印到控制台
    for message in messages:
        print(message)

    # 如果提供了文件路径，则将消息写入该文件
    if file_path:
        try:
            with open(file_path, mode, encoding='utf-8') as f:
                for message in messages:
                    if isinstance(message, (list, tuple, dict)):
                        f.write(str(message))
                    else:
                        f.write(message)
                f.write('\n')
        except Exception as e:
            print(f"无法写入文件: {e}")

# # 示例用法
# custom_print("消息1", "消息2", "消息3", file_path="output.txt", mode='a')


if __name__ == "__main__":
    resultpath = Path("./output/test.txt")
    # Ensure the directory exists
    resultpath.parent.mkdir(parents=True, exist_ok=True)
    custom_print_partial = partial(custom_print, file_path=resultpath, mode='a')
    with resultpath.open('w', encoding='utf-8') as new_file:
        new_file.write("This is a new file created by the script.\n")
    a = [1,2,3,4,5]
    b = (1,2,3,4,5)
    c = {'a':1, 'b':2, 'c':3}
    custom_print_partial(a, b, c)
    custom_print_partial(b)
    custom_print_partial(c)
    # /html/body/div[3]/main/div[2]/div/div/p/text()
    # /html/body/div[3]/main/div[2]/div/div/a[2]