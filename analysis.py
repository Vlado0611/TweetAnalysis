from nltk.util import ngrams
from sklearn.datasets import load_files
import re
from nltk.stem import WordNetLemmatizer # za lematizaciju
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def preprocess():
    dataset = pd.read_csv("Zadatak1.csv")

    temp = dataset.values
    X = np.array(temp[:,3])
    y = temp[:, 4]
    y=y.astype('int')

    lemmatizer = WordNetLemmatizer()

    tweets = []

    for i in range(len(X)):

        tweet = str(X[i])
        tweet = tweet.lower()
        #tweet = re.sub("\W", " ", tweet) # uklanjanje specijalnih karaktera
        tweet = " ".join(tweet.split("\\n")) # mice \\n
        tweet = re.sub("\s[^a-zA-Z]+\s"," ",tweet) # brojeve mijenjamo sa blankom
        tweet = re.sub("[^a-z']+"," ",tweet) # specijalne karaktere mijenjamo sa blankom
        
        tweet = re.sub("\s+"," ",tweet) # sve vezane bjeline mijenjamo se jednim blankom

        words = tweet.split()
        tweet = " ".join([lemmatizer.lemmatize(word) for word in words])
        tweets.append(tweet)

    X = np.array(tweets)
    return X, y

def bagOfWords(X):

    from sklearn.feature_extraction.text import CountVectorizer
    # kaze nam u kom dokumentu se koja rijec javila koliko puta
    vectorizer = CountVectorizer(min_df=10, max_df=0.8,ngram_range=(1,1), stop_words='english')

    X = vectorizer.fit_transform(X).toarray()
    '''
    cv_dataframe=pd.DataFrame(X,columns=vectorizer.get_feature_names())
    print(cv_dataframe)
    '''
    from sklearn.feature_extraction.text import TfidfTransformer
    transformer = TfidfTransformer(use_idf=True, smooth_idf=True)

    X = transformer.fit_transform(X).toarray()
    return X

def nGrams(X, n):
    from sklearn.feature_extraction.text import CountVectorizer
    # kaze nam u kom dokumentu se koja rijec javila koliko puta
    vectorizer = CountVectorizer(max_features=1000, min_df=10, max_df=0.8,ngram_range=(n,n), stop_words='english')

    X = vectorizer.fit_transform(X).toarray()

    from sklearn.feature_extraction.text import TfidfTransformer
    transformer = TfidfTransformer(use_idf=True, smooth_idf=True)

    X = transformer.fit_transform(X).toarray()
    return X

def logReg(X_train, y_train, X_test):
    from sklearn.linear_model import LogisticRegression

    classifier = LogisticRegression(solver="liblinear",class_weight={0:1, 1:2})
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    return y_pred

def SVM(X_train, y_train, X_test):
    from sklearn import svm
    classifier = svm.SVC(gamma=1)
    classifier.fit(X_train,y_train)

    y_pred = classifier.predict(X_test)
    return y_pred

def kNearest(X_train, y_train, X_test):

    from sklearn.neighbors import KNeighborsClassifier
    '''
    error = []
    minerr = 0
    err = 1000
    #Računanje greške za vrijednost K između 1 i 40
    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error.append(np.mean(pred_i != y_test))

        if err > error[i-1]:
            err = error[i-1]
            minerr = i

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
            markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    plt.show()

    k = minerr
    '''
    k = 20

    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train,y_train)

    y_pred = classifier.predict(X_test)
    return y_pred

def decisionTree(X_train, y_train, X_test):
    from sklearn import tree
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    return y_pred

def randomForest(X_train, y_train, X_test):
    from sklearn.ensemble import RandomForestClassifier

    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)


    y_pred = classifier.predict(X_test)
    return y_pred

def naiveBayesG(X_train, y_train, X_test):
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    return y_pred

def naiveBayesM(X_train, y_train, X_test):
    from sklearn.naive_bayes import MultinomialNB
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    
    return y_pred

def report(y_test, y_pred):
    from sklearn.metrics import confusion_matrix, classification_report
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

X, y = preprocess()


#X = nGrams(X, 3)
X = bagOfWords(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

y_pred = naiveBayesM(X_train, y_train, X_test)

report(y_test, y_pred)

