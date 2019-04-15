#Natural Language Processing

#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#Importing Dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)
#delimeter- read csv expecting , but tsv file col separated by tab
#quating = 3 means igmoring "" double quotes, beacuse tsv reveiw cotaining double quotes


#------------------------------------------------Cleaning the text
import re
import nltk
nltk.download('stopwords') #download irrelavant words
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer


corpus = []

for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i],) #1st - remove unneccessary characters
    #removed character replaced with space
    review = review.lower() #converted to lowercase
    
    #split the sentense to  list of words
    review = review.split()
    
    #create porter stammer object
    ps = PorterStemmer()
    
    #remove unnessary words
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    
    review = ' '.join(review)
    corpus.append(review)
    
#----------------------------------------------Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray() #create the metrixes  - sparse metrix
y = dataset.iloc[:,1].values #dependant variable

#create machine learning classification model

#Naive bayes classification model

#Spliting dataset into the Training set and Test 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

#feature scaling is unneccary here.

#Filtering Naive Bayes to the Training set 
from sklearn.naive_bayes import GaussianNB
classifier  = GaussianNB()
classifier.fit(X_train,y_train)

#Prediciting  the Test set result
y_pred = classifier.predict(X_test)

#making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
