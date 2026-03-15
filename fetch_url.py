import urllib.request
import re

try:
    url = "https://openai.com/zh-Hans-CN/index/introducing-gpt-5-3-codex/"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    html = urllib.request.urlopen(req).read().decode('utf-8')
    text = re.sub('<[^<]+?>', ' ', html)
    print(' '.join(text.split())[:2000])
except Exception as e:
    print(f"Error: {e}")
