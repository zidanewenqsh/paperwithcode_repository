import os
import requests
from lxml import etree
from fake_useragent import UserAgent
from pathlib import Path
from tqdm import tqdm

class Spider:
    def __init__(self, logfile, basesavedir="./output"):
        self.logfile = logfile
        self.basesavedir = Path(basesavedir)
        self.basesavedir.mkdir(parents=True, exist_ok=True)
        
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

    def fetch_page(self, url):
        response = requests.get(url, headers=self.headers, proxies=self.proxy)
        if response.status_code == 200:
            return response.text
        else:
            print('请求失败:', response.status_code)
            return None

    def parse_images(self, html):
        tree = etree.HTML(html)
        img_urls = tree.xpath('//img/@src')
        return img_urls

    def download_image(self, img_url, save_dir):
        response = requests.get(img_url, headers=self.headers, proxies=self.proxy)
        if response.status_code == 200:
            img_data = response.content
            img_name = os.path.basename(img_url)
            img_path = save_dir / img_name
            with open(img_path, 'wb') as img_file:
                img_file.write(img_data)
            print(f"Image saved: {img_path}")
        else:
            print(f"Failed to download image: {img_url}")

    def run(self, url):
        html = self.fetch_page(url)
        if html:
            img_urls = self.parse_images(html)
            for img_url in tqdm(img_urls):
                self.download_image(img_url, self.basesavedir)

if __name__ == "__main__":
    spider = Spider(logfile="spider.log", basesavedir="./images")
    target_url = "https://example.com"  # 替换为目标网站的URL
    spider.run(target_url)