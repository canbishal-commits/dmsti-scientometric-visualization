import requests
from bs4 import BeautifulSoup

url = "https://elaba.mb.vu.lt/dmsti/?aut=Martynas+Sabaliauskas"

response = requests.get(url)

soup = BeautifulSoup(response.text, "html.parser")

text = soup.get_text()

lines = text.splitlines()

clean_lines = []

for line in lines:
    line = line.strip()

    if len(line) > 50:
        clean_lines.append(line)

for item in clean_lines[:20]:
    print(item)