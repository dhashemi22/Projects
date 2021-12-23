import sys
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import http.client
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def default_function():
    year1 = 2020
    url21 = "https://www.basketball-reference.com/leagues/NBA_{}_per_game.html".format(year1)
    html21 = urlopen(url21)
    soup21 = BeautifulSoup(html21, features="lxml")
  
    soup21.findAll('tr', limit=2)
    headers21 = [th.getText() for th in soup21.findAll('tr', limit=2)[0].findAll('th')]
    headers21 = headers21[1:]
    
    rows21 = soup21.findAll('tr')[1:]
    player_stats21 = [[td.getText() for td in rows21[i].findAll('td')]
                for i in range(len(rows21))]
    
    stat21 = pd.DataFrame(player_stats21, columns = headers21)
    stats21 = stat21.drop_duplicates(subset=['Player'], keep='first')
    
    
    # Second Dataset with Advanced Statistics
    adv_url21 = "https://www.basketball-reference.com/leagues/NBA_{}_advanced.html".format(year1)
    adv_html21 = urlopen(adv_url21)
    adv_soup21 = BeautifulSoup(adv_html21, features="lxml")
    
    adv_soup21.findAll('tr', limit=2)
    adv_headers21 = [th.getText() for th in adv_soup21.findAll('tr', limit=2)[0].findAll('th')]
    adv_headers21 = adv_headers21[1:]
    
    adv_rows21 = adv_soup21.findAll('tr')[1:]
    adv_player_stats21 = [[td.getText() for td in adv_rows21[i].findAll('td')]
                for i in range(len(adv_rows21))]
    
    adv_stat21 = pd.DataFrame(adv_player_stats21, columns = adv_headers21)
    adv_stats21 = adv_stat21.drop_duplicates(subset=['Player'], keep='first')
    
    
    # Merge both Datasets
    merged_2021 = stats21.merge(adv_stats21, on=['Player', 'Pos', 'Age', 'Tm', 'G'])
    merged_2021.dropna(how='all', inplace = True)
    m21 = merged_2021.drop(merged_2021.columns[[42]], axis=1)
    m21 = m21.reset_index(drop = True)
    m21.to_csv('2021_stats.csv', encoding='utf-8', index = False)
    
    
    
    
    
    # NBA season we will be analyzing
    year2 = 2021
    url22 = "https://www.basketball-reference.com/leagues/NBA_{}_per_game.html".format(year2)
    html22 = urlopen(url22)
    soup22 = BeautifulSoup(html22, features="lxml")
  
    soup22.findAll('tr', limit=2)
    headers22 = [th.getText() for th in soup22.findAll('tr', limit=2)[0].findAll('th')]
    headers22 = headers22[1:]
    
    rows22 = soup22.findAll('tr')[1:]
    player_stats22 = [[td.getText() for td in rows22[i].findAll('td')]
                for i in range(len(rows22))]
    
    stat22 = pd.DataFrame(player_stats22, columns = headers22)
    stats22 = stat22.drop_duplicates(subset=['Player'], keep='first')
    
    
    # Second Dataset with Advanced Statistics
    adv_url22 = "https://www.basketball-reference.com/leagues/NBA_{}_advanced.html".format(year2)
    adv_html22 = urlopen(adv_url22)
    adv_soup22 = BeautifulSoup(adv_html22, features="lxml")
    
    adv_soup22.findAll('tr', limit=2)
    adv_headers22 = [th.getText() for th in adv_soup22.findAll('tr', limit=2)[0].findAll('th')]
    adv_headers22 = adv_headers22[1:]
    
    adv_rows22 = adv_soup22.findAll('tr')[1:]
    adv_player_stats22 = [[td.getText() for td in adv_rows22[i].findAll('td')]
                for i in range(len(adv_rows22))]
    
    adv_stat22 = pd.DataFrame(adv_player_stats22, columns = adv_headers22)
    adv_stats22 = adv_stat22.drop_duplicates(subset=['Player'], keep='first')
    
    
    # Merge both Datasets
    merged_2022 = stats22.merge(adv_stats22, on=['Player', 'Pos', 'Age', 'Tm', 'G'])
    merged_2022.dropna(how='all', inplace = True)
    m22 = merged_2022.drop(merged_2022.columns[[42]], axis=1)
    m22 = m22.reset_index(drop = True)
    m22.to_csv('2022_stats.csv', encoding='utf-8', index = False)
    
    cols = m21.columns
    m21[cols[4:]] = m21[cols[4:]].apply(pd.to_numeric, errors='coerce')
    m21['Age'] = m21['Age'].astype(int)
    print(m21.mean())
    
    cols2 = m22.columns
    m22[cols2[4:]] = m22[cols2[4:]].apply(pd.to_numeric, errors='coerce')
    m22['Age'] = m22['Age'].astype(int)
    print(m22.mean())
    
    
    
if __name__ == '__main__':
    if len(sys.argv) == 1:
    #default mode
        default_function()
