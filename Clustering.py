import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import csv
import os

# Read CSV File
df = pd.read_csv('train_set.csv', sep='\t')
df = df.head(1000)
A = np.array(df)

# Get the content that is going to be vectorized
documents = []
for i in range(A.shape[0]):
    documents.append(A[i,3])

# Vectorize the text
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Apply kmeans and get the clusters
num_clusters = 5
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

cluster_0_count = 0
cluster_0_count_politics = 0
cluster_0_count_film = 0
cluster_0_count_football = 0
cluster_0_count_business = 0
cluster_0_count_technology = 0

cluster_1_count = 0
cluster_1_count_politics = 0
cluster_1_count_film = 0
cluster_1_count_football = 0
cluster_1_count_business = 0
cluster_1_count_technology = 0

cluster_2_count = 0
cluster_2_count_politics = 0
cluster_2_count_film = 0
cluster_2_count_football = 0
cluster_2_count_business = 0
cluster_2_count_technology = 0

cluster_3_count = 0
cluster_3_count_politics = 0
cluster_3_count_film = 0
cluster_3_count_football = 0
cluster_3_count_business = 0
cluster_3_count_technology = 0

cluster_4_count = 0
cluster_4_count_politics = 0
cluster_4_count_film = 0
cluster_4_count_football = 0
cluster_4_count_business = 0
cluster_4_count_technology = 0

# Gets the clusters and check the categories of clustered textx , in order to check the grouping functionality
while documents:
    d = documents.pop()
    c = clusters.pop()
    for i in range(A.shape[0]):
        if A[i, 3] == d:
            if c == 0:
                cluster_0_count +=1
                if A[i, 4] == "Politics":
                    cluster_0_count_politics += 1
                elif A[i, 4] == "Film":
                    cluster_0_count_film += 1
                elif A[i, 4] == "Football":
                    cluster_0_count_football += 1
                elif A[i, 4] == "Business":
                    cluster_0_count_business += 1
                elif A[i, 4] == "Technology":
                    cluster_0_count_technology += 1
            elif c == 1:
                cluster_1_count +=1
                if A[i, 4] == "Politics":
                    cluster_1_count_politics += 1
                elif A[i, 4] == "Film":
                    cluster_1_count_film += 1
                elif A[i, 4] == "Football":
                    cluster_1_count_football += 1
                elif A[i, 4] == "Business":
                    cluster_1_count_business += 1
                elif A[i, 4] == "Technology":
                    cluster_1_count_technology += 1
            elif c == 2:
                cluster_2_count +=1
                if A[i, 4] == "Politics":
                    cluster_2_count_politics += 1
                elif A[i, 4] == "Film":
                    cluster_2_count_film += 1
                elif A[i, 4] == "Football":
                    cluster_2_count_football += 1
                elif A[i, 4] == "Business":
                    cluster_2_count_business += 1
                elif A[i, 4] == "Technology":
                    cluster_2_count_technology += 1
            elif c == 3:
                cluster_3_count +=1
                if A[i, 4] == "Politics":
                    cluster_3_count_politics += 1
                elif A[i, 4] == "Film":
                    cluster_3_count_film += 1
                elif A[i, 4] == "Football":
                    cluster_3_count_football += 1
                elif A[i, 4] == "Business":
                    cluster_3_count_business += 1
                elif A[i, 4] == "Technology":
                    cluster_3_count_technology += 1
            elif c == 4:
                cluster_4_count +=1
                if A[i, 4] == "Politics":
                    cluster_4_count_politics += 1
                elif A[i, 4] == "Film":
                    cluster_4_count_film += 1
                elif A[i, 4] == "Football":
                    cluster_4_count_football += 1
                elif A[i, 4] == "Business":
                    cluster_4_count_business += 1
                elif A[i, 4] == "Technology":
                    cluster_4_count_technology += 1
            A = np.delete(A, i, axis=0)
            break

if not os.path.exists('Output Files/'):
    os.makedirs('Output Files/')

with open('Output Files/clustering_KMeans.csv', 'w') as csv_file:
    f = csv.writer(csv_file, delimiter ='\t', doublequote = 1, quotechar   = '"', escapechar  = None, skipinitialspace = 0, quoting = csv.QUOTE_MINIMAL, lineterminator='\r\n')
    data = [['         ','Politics', 'Film', 'Football', 'Business', 'Technology'],
            ['Cluster 1', (cluster_0_count_politics*1.0)/cluster_0_count, (cluster_0_count_film*1.0)/cluster_0_count, (cluster_0_count_football*1.0)/cluster_0_count, (cluster_0_count_business*1.0)/cluster_0_count, (cluster_0_count_technology*1.0)/cluster_0_count],
            ['Cluster 2', (cluster_1_count_politics*1.0)/cluster_1_count, (cluster_1_count_film*1.0)/cluster_1_count, (cluster_1_count_football*1.0)/cluster_1_count, (cluster_1_count_business*1.0)/cluster_1_count, (cluster_1_count_technology*1.0)/cluster_1_count],
            ['Cluster 3', (cluster_2_count_politics*1.0)/cluster_2_count, (cluster_2_count_film*1.0)/cluster_2_count, (cluster_2_count_football*1.0)/cluster_2_count, (cluster_2_count_business*1.0)/cluster_2_count, (cluster_2_count_technology*1.0)/cluster_2_count],
            ['Cluster 4', (cluster_3_count_politics*1.0)/cluster_3_count, (cluster_3_count_film*1.0)/cluster_3_count, (cluster_3_count_football*1.0)/cluster_3_count, (cluster_3_count_business*1.0)/cluster_3_count, (cluster_3_count_technology*1.0)/cluster_3_count],
            ['Cluster 5', (cluster_4_count_politics*1.0)/cluster_4_count, (cluster_4_count_film*1.0)/cluster_4_count, (cluster_4_count_football*1.0)/cluster_4_count, (cluster_4_count_business*1.0)/cluster_4_count, (cluster_4_count_technology*1.0)/cluster_4_count]]
    f.writerows(data)
result = pd.read_csv('Output Files/clustering_KMeans.csv', sep='\t')
print (result)

