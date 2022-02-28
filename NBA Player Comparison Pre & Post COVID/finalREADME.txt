For the final project...
There is one SCRAPER.PY file to run which scrapes all the data
and outputs simple analysis.
AND
There is one CLUSTER_ANALYSIS.PY file to run which scrapes all the data,
runs Principal Component Analysis, and Clusters players into groups.
There are two static datasets for multiple seasons of play,
Including CSV Files: 2021_stats and 2022_stats
Analysis is also performed on a specific cluster for both files.


Please execute command line:

        'python scraper.py' or
        'python cluster_analysis.py' or
        'python cluster_analysis.py --static'
        in the same path as the scraper.py and cluster_analysis.py files.
        

If you run 'python scraper.py',
I will scrape all the data required from basketball-reference.com, clean and store
the data, then perform some simple analysis on the data as well.

If you run 'python cluster_analysis.py',
I will scrape all the data required from basketball-reference.com, clean and store
the data, the perform PCA in addition to the K-Means Clustering algorithm. Includes
plots and other analysis of clusters.

If you run 'python cluster_analysis.py --static',
I will perform PCA in addition to the K-Means Clustering algorithm from the stored
datasets which have previously been scraped and cleaned. 
NOTE: This direction does NOT scrape data from basketball-reference.com.


==============================================================================

Libraries Used:

sys, urllib.request, bs4, pandas, http.client, json, csv, numpy, matplotlib, seaborn,
sklearn.















