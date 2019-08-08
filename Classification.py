import numpy as np
import pandas
import csv
import os
# Sklearn various libraries
from sklearn.pipeline import Pipeline
from sklearn import model_selection
from sklearn.metrics import make_scorer
from sklearn.preprocessing import LabelBinarizer
# Vectorizers
from sklearn.feature_extraction.text import TfidfVectorizer
# Classification Methods
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
# Metrics
from sklearn.metrics import roc_auc_score

# Get the CSV file
df = pandas.read_csv('train_set.csv', sep='\t')
df = df.head(300)

# Get the content and the categories
X = df['Content'].values
Y = df['Category'].values

# Setup 10 Fold evalution
kfold = model_selection.KFold(n_splits=10, random_state= 7 , shuffle = True)

# Create pipelines for each classification method
NaiveBayes = Pipeline([
    ('vectorizer',  TfidfVectorizer(stop_words='english')),
    ('classifier',  MultinomialNB()) ])

SupportVectorMachines = Pipeline([
    ('vectorizer',  TfidfVectorizer(stop_words='english')),
    ('classifier',  LinearSVC()) ])

RandomForrest = Pipeline([
    ('vectorizer' , TfidfVectorizer(stop_words='english')),
    ('classifier',  RandomForestClassifier(n_estimators = 20) ) ])

# Custom AUC metric score function
def custom_avg_roc_auc_score(truth, pred):
    lb = LabelBinarizer()
    lb.fit(truth)
    truth = lb.transform(truth)
    pred = lb.transform(pred)
    return roc_auc_score(truth, pred, average="macro")

avg_roc_auc_scorer = make_scorer(custom_avg_roc_auc_score)

classification_methods = {'Support Vector Machines' : SupportVectorMachines ,'Naive Bayes':NaiveBayes ,'Random Forests':RandomForrest }
evaluation_methods = {'Recall':'recall_macro','Accuracy': 'accuracy','Precision':'precision_macro','F-Measure':'f1_macro','AUC':avg_roc_auc_scorer}
final_results = {}

# Test each classification method ,
for ckey,cvalue in classification_methods.items():
    for ekey,evalue in evaluation_methods.items():
        # print(ckey + ' ' + ekey)
        results = model_selection.cross_val_score(cvalue, X, Y, cv= kfold, scoring= evalue)
        # print(results)
        result_key = (ckey,ekey)
        final_results[result_key] = (float(sum(results)) / max(len(results), 1))

# print(final_results)

# -----------------------------------------------------------------------------#
# KNN Implementation
# -----------------------------------------------------------------------------#

import collections
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Definition of stopwords
STOPWORDS = (
    "a able about across after all almost also am among an and "
    "any are as at be because been but by can cannot could dear "
    "did do does either else ever every for from get got had has "
    "have he her hers him his how however i if in into is it its "
    "just least let like likely may me might most must my "
    "neither no nor not of off often on only or other our own "
    "rather said say says she should since so some than that the "
    "their them then there these they this tis to too twas us "
    "wants was we were what when where which while who whom why "
    "will with would yet you your".split()
)

# Read CSV file
df = pandas.read_csv('train_set.csv', sep='\t')
df = df.head(100)

names = list(df.columns.values)

X = np.array(df.ix[:, 3])     # end index is exclusive
y = np.array(df['Category'])    # another way of indexing a pandas df

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


def predict(X_train, Y_train, X_test, k):
    distances = []
    targets = []
    for i in range(len(X_train)):
        xtest = X_test.split()
        xtrain = X_train[i].split()
        xtest = [w for w in xtest if not w in STOPWORDS]
        xtrain = [w for w in xtrain if not w in STOPWORDS]
        samewords = len(set(xtest) & set(xtrain))
        if samewords == 0:
            distance = 1000
        else:
            distance = 1.0/samewords
        distances.append([distance, i])
        distances = sorted(distances)
    for i in range(k):
        index = distances[i][1]
        targets.append(Y_train[index])
    return collections.Counter(targets).most_common(1)[0][0]

def kNearestNeighbor(X_train, Y_train, X_test, k):
    predictions = []
    if k > len(X_train):
        raise ValueError
    for i in range(len(X_test)):
        q = predict(X_train, Y_train, X_test[i], k)
        predictions.append(q)
    return predictions

predictions = kNearestNeighbor(X_train, y_train, X_test, 2)
predictions = np.asarray(predictions)

def custom_avg_roc_auc_score(truth, pred):
    lb = LabelBinarizer()
    lb.fit(truth)
    truth = lb.transform(truth)
    pred = lb.transform(pred)
    return roc_auc_score(truth, pred, average="macro")

accuracy = []
precision = []
recall = []
fmeasure = []
auc = []

# Setup 10-Fold for KNN
kfold = model_selection.KFold(n_splits=10, random_state= 7 , shuffle = True)

# For each fold , calculate KNN's accuray , with all different metrics
for (train, test) in kfold.split(X,y):
    predictions = kNearestNeighbor(X[train],y[train],X[test],7)
    # Accuracy Score
    acc = accuracy_score(y[test], predictions)
    # Precision Score
    prec = precision_score(y[test], predictions, average='macro')
    # Recall Score
    rec = recall_score(y[test], predictions, average='macro')
    # F-Measure Score
    f = f1_score(y[test],predictions, average='macro')
    # AUC
    auc_score = custom_avg_roc_auc_score(y[test],predictions)

    auc.append(auc_score)
    accuracy.append(acc)
    precision.append(prec)
    recall.append(rec)
    fmeasure.append(f)

auc = ((float(sum(auc)) / max(len(auc), 1)))
acc = ((float(sum(accuracy)) / max(len(accuracy), 1)))
rec = ((float(sum(recall)) / max(len(recall), 1)))
fm = ((float(sum(fmeasure)) / max(len(fmeasure), 1)))
prec = ((float(sum(precision)) / max(len(precision), 1)))

if not os.path.exists('Output Files'):
    os.makedirs('Output Files')

with open('Output Files/EvaluationMetric_10fold.csv', 'w') as csv_file:
    f = csv.writer(csv_file, delimiter ='\t', doublequote = 1, quotechar   = '"', escapechar  = None, skipinitialspace = 0, quoting = csv.QUOTE_MINIMAL, lineterminator='\r\n')
    data = [['         ','Naive Bayes', 'Random Forest', 'SVM', 'KNN'],
            ['Accuracy' , final_results[('Naive Bayes','Recall')], final_results['Random Forests','Accuracy'], final_results['Support Vector Machines','Accuracy'], acc],
            ['Precision', final_results[('Naive Bayes','Precision')], final_results['Random Forests','Precision'], final_results['Support Vector Machines','Precision'], prec],
            ['Recall   ', final_results[('Naive Bayes','Recall')], final_results['Random Forests','Recall'], final_results['Support Vector Machines','Recall'],rec],
            ['F-Measure', final_results[('Naive Bayes','F-Measure')], final_results['Random Forests','F-Measure'], final_results['Support Vector Machines','F-Measure'],fm],
            ['AUC', final_results[('Naive Bayes','AUC')], final_results['Random Forests','AUC'], final_results['Support Vector Machines','AUC'],auc]]
    f.writerows(data)
result = pandas.read_csv('Output Files/EvaluationMetric_10fold.csv', sep='\t')
print (result)


# Get the content and the categories
X = df['Content'].values
Y = df['Category'].values

# Decision Trees Classification Method
from sklearn import tree
Tree = Pipeline([
    ('vectorizer',  TfidfVectorizer(stop_words='english')),
    ('classifier',  tree.DecisionTreeClassifier()) ])

result = model_selection.cross_val_score(Tree, X, Y, cv= kfold, scoring='accuracy')
# print(result)
