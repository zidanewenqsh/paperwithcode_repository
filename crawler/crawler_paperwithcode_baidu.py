# -*- coding: utf-8 -*-
import requests

url = "https://image.baidu.com/search/acjson"
params = {
    'tn': 'resultjson_com',
    'ipn': 'rj',
    'ct': 201326592,
    'is': '',
    'fp': 'result',
    'queryWord': '猫',
    'cl': 2,
    'lm': -1,
    'ie': 'utf-8',
    'oe': 'utf-8',
    'adpicid': '',
    'st': '',
    'z': '',
    'ic': '',
    'hd': '',
    'latest': '',
    'copyright': '',
    's': '',
    'se': '',
    'tab': '',
    'width': '',
    'height': '',
    'face': '',
    'istype': '',
    'qc': '',
    'nc': 1,
    'expermode': '',
    'nojc': '',
    'isAsync': '',
    'pn': 90,
    'rn': 30,
    'gsm': '5a',
    '1730888268668': ''
}

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
    'referer': 'https://image.baidu.com/search/index?tn=baiduimage&ps=1&ct=201326592&lm=-1&cl=2&nc=1&ie=utf-8&dyTabStr=MCwzLDEsMiwxMyw3LDYsNSwxMiw5&word=%E7%8C%AB',
}

response = requests.get(url, params=params, headers=headers)
data = response.json()
print(data)
