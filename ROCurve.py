import pandas as pd
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
import sklearn.metrics as metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import os
###############################################################################
# Data IO and generation

# import some data to play with

# Import some data to play with
df = pd.read_csv('train_set.csv', sep='\t')
df = df.head(100)

X = df['Content'].values
y = df['Category'].values

###############################################################################
# Classification and ROC analysis

# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=10)
classifier = svm.SVC(kernel='linear', probability=True,
                     random_state=0)

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)

colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
lw = 2

pipeline = Pipeline([
    ('vectorizer',  TfidfVectorizer(stop_words='english')),
    ('classifier',  classifier )])

lb = LabelBinarizer()

n_classes = 5

i = 0

NaiveBayes = Pipeline([
    ('vectorizer',  TfidfVectorizer(stop_words='english')),
    ('classifier',  MultinomialNB()) ])

SupportVectorMachines = Pipeline([
    ('vectorizer',  TfidfVectorizer(stop_words='english')),
    ('classifier',  SVC(probability = True)) ])

RandomForrest = Pipeline([
    ('vectorizer' , TfidfVectorizer(stop_words='english')),
    ('classifier',  RandomForestClassifier(n_estimators = 20) ) ])

classification_methods = {'Support Vector Machines' : SupportVectorMachines ,'Naive Bayes':NaiveBayes ,'Random Forests':RandomForrest }

if not os.path.exists('Output Files/roc_curve10/'):
    os.makedirs('Output Files/roc_curve10/')

for method,pipeline in classification_methods.items():
        print(method)
        fold = 0
        for (train, test) in cv.split(X,y):
            pipeline.fit(X[train],y[train])
            y_score = pipeline.predict_proba(X[test])
            # Compute ROC curve and area the curve
            lb.fit(y[test])
            y_test = lb.transform(y[test])
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test[i], y_score[i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test.ravel(), y_score.ravel())
            roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

            # Plot ROC curves for the multiclass problem

            # Compute macro-average ROC curve and ROC area

            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= n_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(fpr["micro"], tpr["micro"],
                     label='micro-average ROC curve (area = {0:0.2f})'
                           ''.format(roc_auc["micro"]),
                     color='deeppink', linestyle=':', linewidth=4)

            plt.plot(fpr["macro"], tpr["macro"],
                     label='macro-average ROC curve (area = {0:0.2f})'
                           ''.format(roc_auc["macro"]),
                     color='navy', linestyle=':', linewidth=4)

            colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red'])
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                         label='ROC curve of class {0} (area = {1:0.2f})'
                               ''.format(i, roc_auc[i]))

            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Some extension of Receiver operating characteristic to multi-class')
            plt.legend(loc="lower right")

            png = 'Output Files/roc_curve10/' + method + str(fold) + '.png'
            plt.savefig(png)
            plt.close()
            fold += 1
