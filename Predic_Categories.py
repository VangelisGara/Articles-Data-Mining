import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import csv
import os

# Get train set
dftrain = pd.read_csv('train_set.csv', sep='\t')
# Get test set
dftest = pd.read_csv('test_set.csv', sep='\t')

# Create a Naive Bayes Pipeline
pipeline = Pipeline([
    ('vectorizer',  TfidfVectorizer(stop_words='english')),
    ('classifier',  MultinomialNB())])

# Fit train data to Naive Bayes Classifier to work with
pipeline.fit(dftrain['Content'].values ,dftrain['Category'].values)

# Predict test set
predict_test = pipeline.predict(dftest['Content'].values)
predict_test = predict_test.tolist()

ids = dftest['Id'].values
ids = ids.tolist()

# Format predictions
data = [['ID',' ','Predicted_Category']]
while predict_test:
   i = ids.pop()
   c = predict_test.pop()
   data.append([i,' ',c])
   # print(str(i) + ' ' + c)

if not os.path.exists('Output Files'):
    os.makedirs('Output Files')

with open('Output Files/testSet_categories.csv', 'w') as csv_file:
    f = csv.writer(csv_file, delimiter ='\t', doublequote = 1, quotechar   = '"', escapechar  = None, skipinitialspace = 0, quoting = csv.QUOTE_MINIMAL, lineterminator='\r\n')
    f.writerows(data)
result = pd.read_csv('Output Files/testSet_categories.csv', sep='\t')
print (result)



