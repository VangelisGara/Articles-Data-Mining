## Data Mining On Articles
Collecting data, preprocessing, transforming them and applying data mining methods for clustering/classification evaluation.

***train set*** : train_set.csv

It's a tab seberated file, with the following fields, used to train the algorithms:
					

    id: articles id
    titile: articles title
    content: articles content
    category: articles category
***test set*** : test_set.csv

It doesn't contain the content category, cause it's going to be predicted by our classification algorithms

## Wordcloud Creator

A simple word cloud creator, for each article category.

    python3 Wordclouds.py

## Clustering

Implementing clustering on the data with K-Means algorithm and cosine similarity. The algorithm will produce 5 clusters based on the training set, and output the percentage of cluster's data associated to each category.

    python3 Clustering.py
## Classification

Implementing the following classification methods:

 - Support Vector Machines
 - Random Forests
 - Naive Bayes
 - K-Nearest Neighbor
 
 And the following metrics, to 10-cross validate the above methods:
 
 - Precision/ Recall / F-Measure
 - Accuracy
 - AUC
 - ROC plot

On the pre-processing phase we used *Latent Semantic Analysis*.

    python3 Classification.py
    python3 ROCurve.py , to create the ROC plots

To predict a subset of the data, outputting article's ID, and the predicted category for it:

    python3 Predic_Categories.py
    
## Beating the benchmark

In the analysis pdf, you can have a detailed analysis of the algorithms used and the benchmark of each one.

### Important

The implementation used the Sci-kit library mainly for the clustering and classification methods, along with a set of other libraries. 
Make sure to install them by:

    sudo pip3 install -U <library name>


