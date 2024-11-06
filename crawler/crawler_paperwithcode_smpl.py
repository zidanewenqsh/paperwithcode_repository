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
# 数据库连接参数
# db_params = {
#     'host': 'localhost',
#     'user': 'root',
#     'password': 'root',
#     'database': 'test',
#     'charset': 'utf8mb4',
#     'cursorclass': pymysql.cursors.DictCursor
# }
#
# # 连接到数据库
# connection = pymysql.connect(**db_params)
# tablename = 'spider_temp'

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
    def __init__(self, logfile, basesavedir="./output"):
        super().__init__(logfile=logfile)
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
        # self.key = namedtuple('Key', ['i', 'j'])
        # self.value = namedtuple('Value', ['i', 'j'])
        # 发起 HTTP 请求
        # self.response = requests.get(url, headers=headers)
        # self.resultlist = []
        # self.names = set()
        # self.names.add("Tina")
        # self.names.add("Kate")


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


    def resp(self, url, isfile, handle, *args, **kwargs):
        # print(f"{url=}")
        if (isfile):
            assert (os.path.exists("index.html")), "index.html not found"
            with open("index.html", 'r', encoding='utf-8') as f:
                data = f.read()
            # 发起 HTTP 请求
        else:
            response = requests.get(url, headers=self.headers, proxies=self.proxy)
            # 检查请求是否成功
            if response.status_code != 200:
                # print('请求失败:', response.status_code)
                logger.error(f"request failed: {response.status_code=}, {url=}")
                return None
            # baselist = [url]
            # 处理响应数据
            data =response.text
        # print(f"{len(data)=}", data)
        # exit()
        return handle(data, *args, **kwargs)


def treatbase(data, *args, **kwargs):
    # 解析 HTML
    tree = etree.HTML(data)

    
    if len(args) < 2:
        print("args length < 2")
        return
    
    xpath, pattern = args
    assert (isinstance(xpath, str) or isinstance(xpath, list))
    if isinstance(xpath, str):
        hrefs = tree.xpath(xpath)
    else:
        hrefs = [tree.xpath(x) for x in xpath]
        # print(hrefs)
        hrefs = [list(x) for x in zip(*hrefs)]
    if hrefs is None:
        hrefs = []
    # print(hrefs)
    # exit()
    hrefs = [
        (element[2], 'https://paperswithcode.com' + element[0], re.search(pattern, element[1]).group(1))
        for element in hrefs
        if re.search(pattern, element[1])
    ]
    # if pattern is not None or pattern != "":
    #     hrefs = [pattern + item for item in hrefs]
    return hrefs

def treatsubdata(data, *args, **kwargs):
    # 解析 HTML
    tree = etree.HTML(data)

    
    if len(args) < 2:
        print("args length < 2")
        return
    
    xpath, pattern = args
    assert (isinstance(xpath, str) or isinstance(xpath, list))
    if isinstance(xpath, str):
        hrefs = tree.xpath(xpath)
    else:
        hrefs = [tree.xpath(x) for x in xpath]
        # print(hrefs)
        # hrefs = [list(x) for x in zip(*hrefs)]
    if hrefs is None:
        hrefs = []
    # hrefs = [[item.strip() for item in hrefs[0] if item.strip()], hrefs[1], hrefs[2]]
    hrefs = [hrefs[0][0].strip(), hrefs[1][0], hrefs[2]]
    return hrefs
def treatsubdata_code(data, *args, **kwargs):
    # 解析 HTML
    tree = etree.HTML(data)

    
    if len(args) < 2:
        print("args length < 2")
        return
    
    xpath, pattern = args
    assert (isinstance(xpath, str) or isinstance(xpath, list))
    if isinstance(xpath, str):
        hrefs = tree.xpath(xpath)
    else:
        hrefs = [tree.xpath(x) for x in xpath]
        # print(hrefs)
        hrefs = [list(x) for x in zip(*hrefs)]
    if hrefs is None:
        hrefs = []
    # hrefs = [item.strip() for item in hrefs if item.strip()]
    return hrefs

def treatsubdata_dataset(data, *args, **kwargs):
    # 解析 HTML
    tree = etree.HTML(data)

    
    if len(args) < 2:
        print("args length < 2")
        return
    
    xpath, pattern = args
    assert (isinstance(xpath, str) or isinstance(xpath, list))
    if isinstance(xpath, str):
        hrefs = tree.xpath(xpath)
    else:
        hrefs = [tree.xpath(x) for x in xpath]
        # print(hrefs)
        # exit()
        # hrefs = [list(x) for x in zip(*hrefs)]
    if hrefs is None:
        hrefs = []
    hrefs = [
        [item.strip() for item in hrefs[2] if item.strip()], ['https://paperswithcode.com' + x for x in hrefs[0]], hrefs[1],
    ]
    # print(hrefs)
    # for x in hrefs:
    #     print(len(x))
    # exit()
    # exit()
    if len(hrefs[0]) == len(hrefs[1]) == len(hrefs[2]):
        hrefs = [list(x) for x in zip(*hrefs)]
    else:
        return []
    return hrefs

def treatimg(data, *args, **kwargs):
    return data

class Value:
    def __init__(self, i, j):
        self.i = i
        self.j = j

    def __str__(self):
        return f"i:{self.i}-j:{self.j}"

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
    resultpath = Path("./output/result.txt")
    # Ensure the directory exists
    resultpath.parent.mkdir(parents=True, exist_ok=True)
    # custom_print_partial = partial(custom_print)
    custom_print_partial = partial(custom_print, file_path=resultpath, mode='a')

    
    # # Ensure the file exists
    # if not filepath.exists():
    #     filepath.touch()
    # Create a new file and write some initial content
    with resultpath.open('w', encoding='utf-8') as new_file:
        new_file.write("This is a new file created by the script.\n")

    
    spider = Spider(logfile="spider.log", basesavedir="./images")
    target_url = "https://paperswithcode.com/search?q_meta=&q_type=&q=smpl"  # 替换为目标网站的URL
    hrefxpath = "//div[@class='col-lg-3 item-image-col']//a[1]/@href"
    imgxpath = "//div[@class='col-lg-3 item-image-col']//div[@class='item-image']/@style"
    titlexpath = "//div[@class='col-lg-9 item-content']//h1/a/text()"
    # pattern = 'https://paperswithcode.com'
    pattern = r"url\(['\"]?(.*?)['\"]?\)"
    # 调用 Spider 类的方法
    # spider.loadlog()  # 加载日志

    # 获取目标页面并处理
    results = spider.resp(target_url, True, treatbase, [hrefxpath, imgxpath, titlexpath], pattern)
    print(len(results))
    for i, result in enumerate(results):
        print(f"{i=}")
        title, url, imgurl = result
        # print(title, url, imgurl)
        custom_print_partial(f"Title: {title}\n", f"URL: {url}\n", f"Image URL: {imgurl}")

        # exit()
        codexpath1 = "//div[@id='implementations-short-list']//div[@class='paper-impl-cell']//a/@href"
        abstractxpath1 = "/html/body/div[3]/main/div[2]/div/div/p/text()"
        pdfxpath1 = "/html/body/div[3]/main/div[2]/div/div/a[2]/@href"
        # # coderesults = spider.resp(url, False, treatsubdata_code, codexpath1, "")
        # # # print("code", coderesults)
        # # # custom_print("code", coderesults, file_path=resultpath, mode='a')
        # # custom_print_partial("code", coderesults)
        ret = spider.resp(result[1], False, treatsubdata, [abstractxpath1, pdfxpath1, codexpath1], "")
        abstract, pdf, code = ret
        custom_print_partial(f"Abstract: {abstract}\n", f"PDF: {pdf}")
        if len(code) > 0:
            custom_print_partial(f"Code: {code}")
        else:
            print("code is empty")
        # print(ret)
        # exit()
        hrefxpath1 = "//div[@id='datasets']//a[1]/@href "
        imgxpath1 = "//div[@id='datasets']//a/img/@src"
        titlexpath1 = "//div[@id='datasets']//a[contains(@href, 'dataset')]/text()"
        # temp = spider.resp(result[1], False, treatsubdata_temp, titlexpath1, "")
        # print(temp)
        # temp = spider.resp(result[1], False, treatsubdata2, imgxpath1, "")
        # temp = spider.resp(result[1], False, treatsubdata2, hrefxpath1, "")
        ret = spider.resp(result[1], False, treatsubdata_dataset, [hrefxpath1, imgxpath1, titlexpath1], "")
        if len(ret) > 0:
            # print("dataset:", ret)
            # custom_print("dataset:", ret, file_path=resultpath, mode='a')
            custom_print_partial(f"Dataset: {ret}")
        else:
            print("dataset is empty")

        # exit()
        # print("-"*3)
        custom_print_partial("-"*3)
        # exit()
        
    # 保存日志
    # spider.savelog()  # 保存日志到文件
# //*[@id="datasets"]/div/div[2]/div/div/span[1]/a/img
# //div[@id="datasets"]//a[@href="/dataset/human3-6m"]/text()
# //div[@id="datasets"]//a[@href="/dataset/human3-6m"]/text()