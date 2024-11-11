def format_string(s):
    # 将所有字符转换为小写
    s = s.lower()
    # 替换括号为 ' '
    s = s.replace('(', ' ').replace(')', ' ')
    # 替换多个空格为单个空格
    s = ' '.join(s.split())
    # 替换空格为 '-'
    s = s.replace(' ', '-')
    return s

input_string = "Electromyography (EMG)"
output_string = format_string(input_string)
print(output_string)
