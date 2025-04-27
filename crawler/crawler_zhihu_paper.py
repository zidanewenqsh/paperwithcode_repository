# -*- coding: utf-8 -*-
import os

import logging

import os.path
from collections import namedtuple, defaultdict
import requests
from fake_useragent import UserAgent
from lxml import etree
import re
from pathlib import Path
from itertools import repeat
from functools import partial
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
import numpy as np
import pickle
import json
import logging
import pymysql

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

logging.basicConfig(
    level=logging.INFO,  # 设置日志级别，可以是DEBUG、INFO、WARNING、ERROR、CRITICAL
    format='%(asctime)s [%(levelname)s]: %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler('myapp.log')  # 输出到文件
    ]
)
logger = logging.getLogger(__name__)

savedir = "./Saves"
if (not os.path.exists(savedir)):
    os.mkdir(savedir)


class Log:
    def __init__(self, logfile):
        self.logfile = logfile
        self.logdict = {}

    def clearlog(self):
        self.logdict = {}
        if (os.path.exists(self.logfile)):
            self.savelog()
    def savelog(self):
        with open(self.logfile, 'wb') as file:
            pickle.dump(dict(self.logdict), file)

    def loadlog(self):
        if (not os.path.exists(self.logfile)):
            return {}
        with open(self.logfile, 'rb') as file:
            self.logdict = pickle.load(file)

    def record(self, key, value):
        self.logdict[key] = value
        # raise NotImplementedError

class Spider(Log):
    def __init__(self, logfile, basesavedir=None):
        super().__init__(logfile=logfile)
        if basesavedir is not None:
            self.basesavedir = Path(basesavedir)
            self.basesavedir.mkdir(parents=True, exist_ok=True)
        # self.url = url

        # 创建 UserAgent 实例
        ua = UserAgent()

        # 随机生成 User-Agent
        user_agent = ua.random

        # 设置请求头
        self.headers = {
            'User-Agent': user_agent,
        }
        self.proxy = {
            "http": "http://127.0.0.1:7897",
            "https": "http://127.0.0.1:7897",
        }
        self.cache = "./cache"


    def re_init(self):
        # 创建 UserAgent 实例
        ua = UserAgent()
        # 随机生成 User-Agent
        user_agent = ua.random

        # 设置请求头
        self.headers = {
            'User-Agent': user_agent,
        }
        print("re init finished")

    def resp_image(self, imgurl):

        response = requests.get(imgurl, headers=self.headers, proxies=self.proxy)
        if response.status_code == 200:
            # 处理响应数据
            img = response.content
            return img
        else:
            print('请求失败:', response.status_code)
            return None

    def save_cache(self, cachefile, data):
        try:
            with open(str(cachefile), 'w', encoding='utf-8') as f:
                print(data, file=f)
        except Exception as e:
            logger.error(f"{cachefile=}")
            return
    def load_cache(self, cachefile):
        with open(str(cachefile), 'r', encoding='utf-8') as f:
            data = f.read()
        return data
    def get_cache_file(self, url):
        return Path(self.cache)/url.replace("https://", "").replace("\\", "_").replace("/", "_")

    def resp(self, url, data, handle, *args, **kwargs):
        assert url is None or data is None, "url and data cannot be both non-None"
        if data is None:
            try:
                response = requests.get(url, headers=self.headers, proxies=self.proxy)
                # 检查请求是否成功
                if response.status_code != 200:
                    # print('请求失败:', response.status_code)
                    logger.error(f"request failed: {response.status_code=}, {url=}")
                    return None
                # baselist = [url]
                # 处理响应数据
                data =response.text
            except Exception as e:
                logger.error(f"{e=}")
                logger.error(f"{url=}")
                return None
        return handle(data, *args, **kwargs)
        

def readhtml(htmlfile):
    with open(htmlfile, 'r', encoding='utf-8') as f:
        data = f.read()
    return data


def getdata(data):
    return data

def treat(data, *args, **kwargs):
    # 解析 HTML
    tree = etree.HTML(data)

    
    # if len(args) < 2:
    #     print("args length < 2")
    #     return
    
    xpath = args[0]

    assert (isinstance(xpath, str) or isinstance(xpath, list))
    if isinstance(xpath, str):
        hrefs = tree.xpath(xpath)
    else:
        hrefs = [tree.xpath(x) for x in xpath]
        # print(hrefs)
        hrefs = [list(x) for x in zip(*hrefs)]
    if hrefs is None:
        hrefs = {}
    return hrefs

def treatbase(data, *args, **kwargs):
    hrefs = treat(data, *args, **kwargs)
    hrefs = [item.strip() for item in hrefs]
    return hrefs

def treatsubdata(data, *args, **kwargs):
    hrefs = treat(data, *args, **kwargs)
    # print(type(hrefs))
    hrefs = ["https://paperswithcode.com" + item for item in hrefs]
    return hrefs

def treatsubdata2(data, *args, **kwargs):
    hrefs = treat(data, *args, **kwargs)
    # print(type(hrefs))
    hrefs = [[item[0], "https://paperswithcode.com" + item[1], item[2]] for item in hrefs]
    return hrefs

def treatimg(data, *args, **kwargs):
    return data


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
    spider = Spider(logfile="spider.log", basesavedir="./images")
    title_xpath = '//*[@id="ariaTipText"]/@aria-label'
    # contentxpath = '//*[@id="root"]/div/main/div/div/div[3]/div[1]/div/div[3]/div/div/div/div[2]/span[1]/div/div/span/p/text()'
    contentxpath = '//*[@id="root"]/div/main/div/div/div[3]/div[1]/div/div[3]/div/div[2]/div/div/div[2]/span[1]/div/div/span/p/text()'
    # '//*[@id="root"]/div/main/div/div/div[3]/div[1]/div/div[2]/div/div/div/div[3]/span[1]/div/div/span/p[3]'
    data = readhtml("zhihu.html")
    title = spider.resp(None, data, treatbase, title_xpath)[0]
    
    # 使用正则表达式提取问题标题 - 更精确的匹配
    pattern = r"欢迎进入\s+(.*?)\s+-\s+知乎"
    match = re.search(pattern, title)
    if match:
        question_title = match.group(1)
        print("提取的问题标题:", question_title)
    else:
        # 备用方案 - 直接截取前后部分
        if "欢迎进入" in title and "- 知乎" in title:
            start_idx = title.index("欢迎进入") + len("欢迎进入")
            end_idx = title.index("- 知乎")
            question_title = title[start_idx:end_idx].strip()
            print("提取的问题标题:", question_title)
        else:
            print("未能提取到问题标题")
            print("原标题:", title)
    
    # exit()
    results = spider.resp(None, data, treatbase, contentxpath)
    all_results = "".join(results)
    # print(results)
    print(question_title)
    # exit()
    # 将 results 列表中的所有内容合并并写入文件
    with open(f"{question_title}.md", "w", encoding="utf-8") as f:
        # print(all_results, file=f)
        for x in results:
            print(x, file=f)


