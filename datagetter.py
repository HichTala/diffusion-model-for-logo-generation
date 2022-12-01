import csv
import requests
from tqdm import tqdm
import os
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
from PIL import Image
import io


class DatasetGetter():
    def __init__(self, targets, dataset='dataset.csv'):
        self.dataset = dataset
        self.targets = targets

    def scrap(self):
        '''
            Generates a csv file with logo urls and corresponding captions
        ''' 
        print('Scraping is on its way !')
        with open(self.targets, 'r') as f:
            targets = f.read().splitlines()
        try:
            for target in tqdm(targets, colour='cyan'):
                html_doc = requests.get(target).content
                soup = BeautifulSoup(html_doc, 'html.parser')

                url = []
                description = []

                cards = soup.select('div[class*=x-cell]')
                for card in cards:
                    try:
                        url.append(card.find('img').get('data-src'))
                        description.append(card.find('p').text)
                    except AttributeError as e:
                        pass
                url = pd.Series(url)
                description = pd.Series(description)
                dataset = {'url': url, 'description': description}
                dataset = pd.concat(dataset, axis=1)
                dataset.drop_duplicates()
                dataset.to_csv('dataset.csv', mode='a',
                               index=False, header=False, sep=';')
        except requests.exceptions.ConnectionError as e:
            print('Not connected to the internet, check for connection and restart')

    def download(self, delete_tmp_folder=False):
        '''
            Downloads images and export caption into a .tar file from the dataset
        '''
        print('Logo downloading is going on !')
        Path('dataset').mkdir(exist_ok=True)
        with open(self.dataset, 'r', encoding='utf8') as f:
            filereader = csv.reader(f, delimiter=';')
            for i, row in enumerate(tqdm(filereader)):
                url = row[0]
                caption = row[1]
                img_data = requests.get(url).content

                try:
                    img = Image.open(io.BytesIO(img_data)).convert('RGBA')
                    new_image = Image.new('RGBA', img.size, 'WHITE')
                    new_image.paste(img, mask=img)

                    new_image.convert('RGB').save('dataset/' + str(i) + '.jpg')

                    with open('dataset/' + str(i)+'.txt', 'w', encoding='utf8') as captions:
                        captions.write(caption)
                    with open('clip_dataset.csv', 'a', encoding='utf8') as clip_dataset:
                        clip_dataset.write('dataset/' + str(i) + '.jpg;' + caption + '\n')
                except:
                    print('Image ' + str(i) + ' : encoding error, skipped')
        os.system('tar -cvzf dataset.tar -C dataset .')
        if delete_tmp_folder:
            os.system('rm -r dataset')

imagedownloader = DatasetGetter('targets.txt')
# imagedownloader.scrap()
imagedownloader.download()
