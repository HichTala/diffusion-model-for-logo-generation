import pandas as pd
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup

with open('targets.txt', 'r') as f:
    targets = f.read().splitlines()

for target in tqdm(targets, colour="cyan"):
    html_doc = requests.get(target).content
    soup = BeautifulSoup(html_doc, 'html.parser')

    url = []
    description = []

    cards = soup.select("div[class*=x-cell]")
    for card in cards:
        try:
            url.append(card.find('img').get('data-src'))
            description.append(card.find('p').text)
        except AttributeError as e:
            pass

    url = pd.Series(url)
    description = pd.Series(description)

    dataset = {"url": url, "description": description}
    dataset = pd.concat(dataset, axis=1)

    dataset.drop_duplicates()
    dataset.to_csv('dataset.csv', mode='a', index=False, header=False)
