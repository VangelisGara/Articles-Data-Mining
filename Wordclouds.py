from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('train_set.csv',sep = '\t') #CSV ~> Dataframe
A = np.array(df) #Dataframe ~> Python Array

STOPWORDS.add('now')
STOPWORDS.add('will')
STOPWORDS.add('say')
STOPWORDS.add('said')
STOPWORDS.add('people')
STOPWORDS.add('on')
STOPWORDS.add('one')
STOPWORDS.add('also')
STOPWORDS.add('says')
STOPWORDS.add('saying')
STOPWORDS.add('even')

#Creates a word cloud given a text
def wc(word_string):
   wordcloud = WordCloud(
                       stopwords=STOPWORDS,
                       background_color='white',
                       ).generate(word_string)
   plt.imshow(wordcloud)
   plt.axis('off')
   plt.show()

#Sums up the contend of each category's article , in order to create word clouds   
wcpol = ''
wcfil = ''
wcfot = ''
wcbus = ''
wctec = ''
for i in range(A.shape[0]):
   category = str(A[i,4])
   if  category == 'Politics':
      wcpol += str(A[i,3]) + ","
   if  category == 'Film':
      wcfil += str(A[i,3]) + ","
   if  category == 'Football':
      wcfot += str(A[i,3]) + ","
   if  category == 'Business':
      wcbus += str(A[i,3]) + ","
   if  category == 'Technology':
      wctec += str(A[i,3]) + ","

wc(wcpol)
wc(wcfil)
wc(wcfot)
wc(wcbus)
wc(wctec)

