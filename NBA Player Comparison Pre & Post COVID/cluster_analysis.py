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
    m21.mean()
    
    cols2 = m22.columns
    m22[cols2[4:]] = m22[cols2[4:]].apply(pd.to_numeric, errors='coerce')
    m22['Age'] = m22['Age'].astype(int)
    m22.mean()
    
    
    
    m21_players = m21['Player']
    m21 = m21.iloc[: , 4:]

    m21 = m21.fillna(0)

    scaler = StandardScaler()
    m21_std = scaler.fit_transform(m21)

    pca = PCA()
    pca.fit(m21_std)

    plt.figure(figsize = (10,8))
    plt.plot(range(1,47), pca.explained_variance_ratio_.cumsum(), marker = 'o', linestyle = '--')
    plt.title('Explained Variance by Components: 2019-2020 Season')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.show()
    
    
    
    pca = PCA(n_components = 7)
    
    pca.fit(m21_std)
    
    pca.transform(m21_std)
    
    scores_pca = pca.transform(m21_std)
    
    wcss = []
    for i in range(1,21):
        kmeans_pca = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans_pca.fit(scores_pca)
        wcss.append(kmeans_pca.inertia_)
          
    
        
    plt.figure(figsize = (10,8))
    plt.plot(range(1,21), wcss, marker = 'o', linestyle = '--')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('K-means with PCA Clustering: 2019-2020 Season')
    plt.show()
    
    
    kmeans_pca = KMeans(n_clusters = 6, init = 'k-means++', random_state = 42)
    
    kmeans_pca.fit(scores_pca)
    
    m21_seg_pca_kmeans = pd.concat([m21.reset_index(drop = True), pd.DataFrame(scores_pca)], axis = 1)
    m21_seg_pca_kmeans.columns.values[-7:] = ['Component1', 'Component2', 'Component3', 'Component4', 'Component5', 'Component6', 'Component7']
    m21_seg_pca_kmeans['Segment K-means PCA'] = kmeans_pca.labels_
    
    m21_seg_pca_kmeans['Segment'] = m21_seg_pca_kmeans['Segment K-means PCA'].map({0:'first',
                                                                                1:'second',
                                                                                2:'third',
                                                                                3:'fourth',
                                                                                4:'fifth',
                                                                                5:'sixth'})
    m21_players = m21_players.reset_index(drop = True)
    m21_seg_pca_kmeans['Player'] = m21_players

    x_axis = m21_seg_pca_kmeans['Component2']
    y_axis = m21_seg_pca_kmeans['Component1']
    plt.figure(figsize = (20, 10))
    sns.scatterplot(x_axis, y_axis, hue = m21_seg_pca_kmeans['Segment'], palette = ['g', 'r', 'c', 'm', 'black', 'pink'])
    plt.title('Clusters by PCA Components: 2019-2020 Season')
    
    plt.show()
        
    cluster_1 = m21_seg_pca_kmeans[m21_seg_pca_kmeans.Segment == 'first']
    cluster_2 = m21_seg_pca_kmeans[m21_seg_pca_kmeans.Segment == 'second']
    cluster_3 = m21_seg_pca_kmeans[m21_seg_pca_kmeans.Segment == 'third']
    cluster_4 = m21_seg_pca_kmeans[m21_seg_pca_kmeans.Segment == 'fourth']
    cluster_5 = m21_seg_pca_kmeans[m21_seg_pca_kmeans.Segment == 'fifth']
    cluster_6 = m21_seg_pca_kmeans[m21_seg_pca_kmeans.Segment == 'sixth']
    
    print(cluster_2)
    
    
    
    
    
    
    m22_players = m22['Player']
    m22 = m22.iloc[: , 4:]
    
    m22 = m22.fillna(0)
    
    scaler2 = StandardScaler()
    m22_std = scaler2.fit_transform(m22)
    
    pca2 = PCA()
    pca2.fit(m22_std)
    
    plt.figure(figsize = (10,8))
    plt.plot(range(1,47), pca2.explained_variance_ratio_.cumsum(), marker = 'o', linestyle = '--')
    plt.title('Explained Variance by Components: 2020-2021 Season')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    
    
    pca2 = PCA(n_components = 7)
    
    pca2.fit(m22_std)
    
    pca2.transform(m22_std)
    
    scores_pca2 = pca2.transform(m22_std)
    
    wcss2 = []
    for i in range(1,21):
        kmeans_pca2 = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans_pca2.fit(scores_pca2)
        wcss2.append(kmeans_pca2.inertia_)
        
    
        
    plt.figure(figsize = (10,8))
    plt.plot(range(1,21), wcss2, marker = 'o', linestyle = '--')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('K-means with PCA Clustering: 2020-2021 Season')
    plt.show()
    
    
    kmeans_pca2 = KMeans(n_clusters = 6, init = 'k-means++', random_state = 42)
    
    kmeans_pca2.fit(scores_pca2)
    
    m22_seg_pca_kmeans = pd.concat([m22.reset_index(drop = True), pd.DataFrame(scores_pca2)], axis = 1)
    m22_seg_pca_kmeans.columns.values[-7:] = ['Component1', 'Component2', 'Component3', 'Component4', 'Component5', 'Component6', 'Component7']
    m22_seg_pca_kmeans['Segment K-means PCA'] = kmeans_pca2.labels_

    m22_seg_pca_kmeans['Segment'] = m22_seg_pca_kmeans['Segment K-means PCA'].map({0:'first',
                                                                                1:'second',
                                                                                2:'third',
                                                                                3:'fourth',
                                                                                4:'fifth',
                                                                                5:'sixth'})
    m22_players = m22_players.reset_index(drop = True)
    m22_seg_pca_kmeans['Player'] = m22_players
        
    x_axis2 = m22_seg_pca_kmeans['Component2']
    y_axis2 = m22_seg_pca_kmeans['Component1']
    plt.figure(figsize = (20, 10))
    sns.scatterplot(x_axis2, y_axis2, hue = m22_seg_pca_kmeans['Segment'], palette = ['g', 'r', 'c', 'm', 'black', 'pink'])
    plt.title('Clusters by PCA Components: 2020-2021 Season')

    plt.show()
    
    m22_cluster_1 = m22_seg_pca_kmeans[m22_seg_pca_kmeans.Segment == 'first']
    m22_cluster_2 = m22_seg_pca_kmeans[m22_seg_pca_kmeans.Segment == 'second']
    m22_cluster_3 = m22_seg_pca_kmeans[m22_seg_pca_kmeans.Segment == 'third']
    m22_cluster_4 = m22_seg_pca_kmeans[m22_seg_pca_kmeans.Segment == 'fourth']
    m22_cluster_5 = m22_seg_pca_kmeans[m22_seg_pca_kmeans.Segment == 'fifth']
    m22_cluster_6 = m22_seg_pca_kmeans[m22_seg_pca_kmeans.Segment == 'sixth']
    
    print(m22_cluster_3)
    
    m21_stars = list(cluster_2['Player'])
    m22_stars = list(m22_cluster_4['Player'])
    list(set(m21_stars) - set(m22_stars))
    
    cluster_2.mean()
    m22_cluster_4.mean()
    
    
    # Percentages of Games Missed by Top-Tier Players
    # 2019-2020 Season
    cluster_2['G'].sum()
    index2 = cluster_2.index
    number_of_rows2 = len(index2)
    number_of_rows2 * 70
    percentage_games_missed2 = (4060-3382)/4060
    percentage_games_missed2*100
    
    
    # 2020-2021 Season
    m22_cluster_4['G'].sum()
    index = m22_cluster_4.index
    number_of_rows = len(index)
    number_of_rows * 72
    percentage_games_missed = (2880-2329)/2880
    percentage_games_missed*100
   
   
    
def static_function():
#static functions
#add the functions/code you need to open and print the static copies of your data
#this function will be run when you type into the command line
    
    m21 = pd.read_csv('2021_stats.csv')
    m22 = pd.read_csv('2022_stats.csv')
    cols = m21.columns
    m21[cols[4:]] = m21[cols[4:]].apply(pd.to_numeric, errors='coerce')
    m21['Age'] = m21['Age'].astype(int)
    m21.mean()
    
    cols2 = m22.columns
    m22[cols2[4:]] = m22[cols2[4:]].apply(pd.to_numeric, errors='coerce')
    m22['Age'] = m22['Age'].astype(int)
    m22.mean()
    
    
    
    m21_players = m21['Player']
    m21 = m21.iloc[: , 4:]

    m21 = m21.fillna(0)

    scaler = StandardScaler()
    m21_std = scaler.fit_transform(m21)

    pca = PCA()
    pca.fit(m21_std)

    plt.figure(figsize = (10,8))
    plt.plot(range(1,47), pca.explained_variance_ratio_.cumsum(), marker = 'o', linestyle = '--')
    plt.title('Explained Variance by Components: 2019-2020 Season')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.show()
    
    
    
    pca = PCA(n_components = 7)
    
    pca.fit(m21_std)
    
    pca.transform(m21_std)
    
    scores_pca = pca.transform(m21_std)
    
    wcss = []
    for i in range(1,21):
        kmeans_pca = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans_pca.fit(scores_pca)
        wcss.append(kmeans_pca.inertia_)
          
    
        
    plt.figure(figsize = (10,8))
    plt.plot(range(1,21), wcss, marker = 'o', linestyle = '--')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('K-means with PCA Clustering: 2019-2020 Season')
    plt.show()
    
    
    kmeans_pca = KMeans(n_clusters = 6, init = 'k-means++', random_state = 42)
    
    kmeans_pca.fit(scores_pca)
    
    m21_seg_pca_kmeans = pd.concat([m21.reset_index(drop = True), pd.DataFrame(scores_pca)], axis = 1)
    m21_seg_pca_kmeans.columns.values[-7:] = ['Component1', 'Component2', 'Component3', 'Component4', 'Component5', 'Component6', 'Component7']
    m21_seg_pca_kmeans['Segment K-means PCA'] = kmeans_pca.labels_
    
    m21_seg_pca_kmeans['Segment'] = m21_seg_pca_kmeans['Segment K-means PCA'].map({0:'first',
                                                                                1:'second',
                                                                                2:'third',
                                                                                3:'fourth',
                                                                                4:'fifth',
                                                                                5:'sixth'})
    m21_players = m21_players.reset_index(drop = True)
    m21_seg_pca_kmeans['Player'] = m21_players

    x_axis = m21_seg_pca_kmeans['Component2']
    y_axis = m21_seg_pca_kmeans['Component1']
    plt.figure(figsize = (20, 10))
    sns.scatterplot(x_axis, y_axis, hue = m21_seg_pca_kmeans['Segment'], palette = ['g', 'r', 'c', 'm', 'black', 'pink'])
    plt.title('Clusters by PCA Components: 2019-2020 Season')
    
    plt.show()
        
    cluster_1 = m21_seg_pca_kmeans[m21_seg_pca_kmeans.Segment == 'first']
    cluster_2 = m21_seg_pca_kmeans[m21_seg_pca_kmeans.Segment == 'second']
    cluster_3 = m21_seg_pca_kmeans[m21_seg_pca_kmeans.Segment == 'third']
    cluster_4 = m21_seg_pca_kmeans[m21_seg_pca_kmeans.Segment == 'fourth']
    cluster_5 = m21_seg_pca_kmeans[m21_seg_pca_kmeans.Segment == 'fifth']
    cluster_6 = m21_seg_pca_kmeans[m21_seg_pca_kmeans.Segment == 'sixth']
    
    print(cluster_2)
    
    
    
    
    
    
    m22_players = m22['Player']
    m22 = m22.iloc[: , 4:]
    
    m22 = m22.fillna(0)
    
    scaler2 = StandardScaler()
    m22_std = scaler2.fit_transform(m22)
    
    pca2 = PCA()
    pca2.fit(m22_std)
    
    plt.figure(figsize = (10,8))
    plt.plot(range(1,47), pca2.explained_variance_ratio_.cumsum(), marker = 'o', linestyle = '--')
    plt.title('Explained Variance by Components: 2020-2021 Season')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    
    
    pca2 = PCA(n_components = 7)
    
    pca2.fit(m22_std)
    
    pca2.transform(m22_std)
    
    scores_pca2 = pca2.transform(m22_std)
    
    wcss2 = []
    for i in range(1,21):
        kmeans_pca2 = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans_pca2.fit(scores_pca2)
        wcss2.append(kmeans_pca2.inertia_)
        
    
        
    plt.figure(figsize = (10,8))
    plt.plot(range(1,21), wcss2, marker = 'o', linestyle = '--')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('K-means with PCA Clustering: 2020-2021 Season')
    plt.show()
    
    
    kmeans_pca2 = KMeans(n_clusters = 6, init = 'k-means++', random_state = 42)
    
    kmeans_pca2.fit(scores_pca2)
    
    m22_seg_pca_kmeans = pd.concat([m22.reset_index(drop = True), pd.DataFrame(scores_pca2)], axis = 1)
    m22_seg_pca_kmeans.columns.values[-7:] = ['Component1', 'Component2', 'Component3', 'Component4', 'Component5', 'Component6', 'Component7']
    m22_seg_pca_kmeans['Segment K-means PCA'] = kmeans_pca2.labels_

    m22_seg_pca_kmeans['Segment'] = m22_seg_pca_kmeans['Segment K-means PCA'].map({0:'first',
                                                                                1:'second',
                                                                                2:'third',
                                                                                3:'fourth',
                                                                                4:'fifth',
                                                                                5:'sixth'})
    m22_players = m22_players.reset_index(drop = True)
    m22_seg_pca_kmeans['Player'] = m22_players
        
    x_axis2 = m22_seg_pca_kmeans['Component2']
    y_axis2 = m22_seg_pca_kmeans['Component1']
    plt.figure(figsize = (20, 10))
    sns.scatterplot(x_axis2, y_axis2, hue = m22_seg_pca_kmeans['Segment'], palette = ['g', 'r', 'c', 'm', 'black', 'pink'])
    plt.title('Clusters by PCA Components: 2020-2021 Season')

    plt.show()
    
    m22_cluster_1 = m22_seg_pca_kmeans[m22_seg_pca_kmeans.Segment == 'first']
    m22_cluster_2 = m22_seg_pca_kmeans[m22_seg_pca_kmeans.Segment == 'second']
    m22_cluster_3 = m22_seg_pca_kmeans[m22_seg_pca_kmeans.Segment == 'third']
    m22_cluster_4 = m22_seg_pca_kmeans[m22_seg_pca_kmeans.Segment == 'fourth']
    m22_cluster_5 = m22_seg_pca_kmeans[m22_seg_pca_kmeans.Segment == 'fifth']
    m22_cluster_6 = m22_seg_pca_kmeans[m22_seg_pca_kmeans.Segment == 'sixth']
    
    print(m22_cluster_4)
    
    m21_stars = list(cluster_2['Player'])
    m22_stars = list(m22_cluster_4['Player'])
    list(set(m21_stars) - set(m22_stars))
    
    cluster_2.mean()
    m22_cluster_4.mean()
    
    
    # Percentages of Games Missed by Top-Tier Players
    # 2019-2020 Season
    cluster_2['G'].sum()
    index2 = cluster_2.index
    number_of_rows2 = len(index2)
    number_of_rows2 * 70
    percentage_games_missed2 = (4060-3382)/4060
    percentage_games_missed2*100
    
    
    # 2020-2021 Season
    m22_cluster_4['G'].sum()
    index = m22_cluster_4.index
    number_of_rows = len(index)
    number_of_rows * 72
    percentage_games_missed = (2880-2329)/2880
    percentage_games_missed*100
    
    
if __name__ == '__main__':
    if len(sys.argv) == 1:
    #default mode
        default_function()
    elif sys.argv[1] == '--static':
    #static mode
        static_function()
    
    
    
    