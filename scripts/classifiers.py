import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#plt.style.use("ggplot")
#plt.style.use("fivethirtyeight")
from cycler import cycler
from itertools import chain
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn import metrics
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,Dense, Dropout, Masking, Embedding
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score,KFold, StratifiedKFold,train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import FreqDist, bigrams
from time import perf_counter_ns
from sklearn.datasets import make_blobs
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA

def mnb_classify(sets,datanv): 
    #return divide_dataset(data_vect,labels,data_vect_t,labels_t,data_e,labels_e),data_t
    #X=np.concatenate((sets[0],sets[1]))
    #y=np.concatenate((sets[2],sets[3]))    
    X=sets[0]
    y=sets[1]
    start_time=perf_counter_ns()
    mnb = MultinomialNB()
    scores = cross_val_score(mnb, X, y, scoring='accuracy', cv=StratifiedKFold(n_splits=10), n_jobs=-1)
    print('mnb Accuracy with cross validation: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    mnb.fit(sets[4],sets[5])
    y_pred = mnb.predict(sets[2])
    y_list=y_pred.tolist()
    print("MNB Accuracy:",metrics.accuracy_score(sets[3], y_pred))
    print(metrics.classification_report(sets[3], y_pred))
    matBinary = metrics.confusion_matrix(sets[3], y_pred)
    print(confusion_matrix(sets[3], y_pred))
    end_time=perf_counter_ns()
    with open("pred_mnb.csv", "w") as file_csv:
        file_csv.write("predicted,label,value"+"\n")
        for i in range (0,len(y_list)):
            file_csv.write(str(y_list[i])+","+str(sets[3][i])+","+str(datanv[i])+"\n")

    print("MNB time to train: ",(end_time-start_time)/1000000000," seconds")


def svm_classify_l(sets,datanv):
    #sets=[data_vect,labels,data_vect_t,labels_t,data_e,labels_e]
    X=sets[0]
    y=sets[1]
    start_time=perf_counter_ns()
    model = SVC(kernel="linear")
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=StratifiedKFold(n_splits=5), n_jobs=-1)
    print('svm linear Accuracy with cross validation: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    expected = sets[3]
    model.fit(sets[4],sets[5])
    predicted = model.predict(sets[2])
    y_list=predicted.tolist()
    with open("pred_svm.csv", "w") as file_csv:
        file_csv.write("predicted,label,value"+"\n")
        for i in range (0,len(y_list)):
            file_csv.write(str(y_list[i])+","+str(sets[3][i])+","+str(datanv[i])+"\n")
    print("SVM linear Accuracy:",metrics.accuracy_score(sets[3], predicted))
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    end_time=perf_counter_ns()
    print("SVM linear time to train: ",(end_time-start_time)/1000000000," seconds")
    
    
def svm_classify_r(sets,datanv):
    X=sets[0]
    y=sets[1]
    start_time=perf_counter_ns()
    model = SVC(kernel="rbf")
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=StratifiedKFold(n_splits=5), n_jobs=-1)
    print('svm rbf Accuracy with cross validation: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    expected = sets[3]
    model.fit(sets[4],sets[5])
    predicted = model.predict(sets[2])
    y_list=predicted.tolist()
    with open("pred_svm.csv", "w") as file_csv:
        file_csv.write("predicted,label,value"+"\n")
        for i in range (0,len(y_list)):
            file_csv.write(str(y_list[i])+","+str(sets[3][i])+","+str(datanv[i])+"\n")
    print("SVM rbf Accuracy:",metrics.accuracy_score(sets[3], predicted))
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    end_time=perf_counter_ns()
    print("SVM rbf time to train: ",(end_time-start_time)/1000000000," seconds")

def FFNN_classify(sets):
    X=sets[0]
    y=sets[1]
    #X=sets[4]
    #y=sets[5]
    num_rows, num_cols = X.shape
    num_words=num_cols
    FNN_start_time=perf_counter_ns()
    model = Sequential()
    model.add(Dense(64, input_shape=(num_words,), activation="sigmoid"))
    model.add(Dense(32, activation="sigmoid"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    history=model.fit(X, y, epochs=10, batch_size=10)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['accuracy'])
    plt.show()
    accuracy = model.evaluate(sets[2], sets[3])
    print("FFNN Accuracy: ",accuracy)
    FNN_end_time=perf_counter_ns()
    print("FFNN time to train: ",(FNN_end_time-FNN_start_time)/1000000000," seconds")
    return accuracy

def k_cross_val(sets):
    metrics=[]
    accuracy=[]
    loss=[]
    cv=StratifiedKFold(n_splits=5)
    X=sets[0]
    y=sets[1]
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        splitted_sets=[X_train,y_train,X_test,y_test]
        metrics.append(FFNN_classify(splitted_sets))
    accuracy= [a[1] for a in metrics]
    loss= [a[0] for a in metrics]
    print(accuracy)
    print('FFNN Accuracy with cross validation: %.3f (%.3f)' % (np.mean(accuracy), np.std(accuracy)))
    print('loss: %.3f (%.3f)' % (np.mean(loss), np.std(loss)))

def divide_dataset_random(data,labels):
    X_train, X_test, y_train, y_test = train_test_split(
           data,labels,
           test_size = 0.25, random_state = 42)
    return[X_train,X_test,y_train,y_test,data]

def divide_dataset(data,labels,data_t,labels_t,e1,e2):
    return[data,labels,data_t,labels_t,e1,e2]

def vectorize(column):
    #vectorizer = CountVectorizer()
    vectorizer=CountVectorizer(ngram_range=(1, 2))
    vectorizer.fit(column)
    vector = vectorizer.transform(column)
    return vector.toarray()

def vectorize_tfidf(column):
    start_time=perf_counter_ns()
    vectorizer = TfidfVectorizer(ngram_range=(1, 1))
    vector = vectorizer.fit_transform(column)
    feature_names = vectorizer.get_feature_names()
    dense = vector.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)
    end_time=perf_counter_ns()
    print("time to transform: ",(end_time-start_time)/1000000000," seconds")
    return df.values

def transform(filename):
    filem=r'{0}.csv'.format(filename)
    data_frame=pd.read_csv(filem,encoding='latin-1')
    misogino = data_frame["misogino"]
    no_misogino = data_frame["no_misogino"]
    misogino_t = data_frame["mis_train"].dropna()
    no_misogino_t = data_frame["nomis_train"].dropna()
    maxwords(misogino)
    maxwords(no_misogino)
    data = list(chain(misogino, misogino_t,no_misogino,no_misogino_t))
    data_t = list(chain(misogino_t, no_misogino_t))
    labels = np.concatenate((np.ones(len(misogino)+(len(misogino_t))),np.zeros(len(no_misogino)+len(no_misogino_t))))
    labels_t = np.concatenate((np.ones(len(misogino_t)),np.zeros(len(no_misogino_t))))
    data_vect=vectorize_tfidf(data)
    data_vect_t=np.concatenate((data_vect[850:1100],data_vect[-250:]))
    data_e = np.concatenate((data_vect[0:850],data_vect[1100:1950]))
    labels_e = np.concatenate((np.ones(len(misogino)),np.zeros(len(no_misogino))))
    sets=[data_vect,labels,data_vect_t,labels_t,data_e,labels_e]
    return sets,data_t

def plot(matBinary):
    labels = ['negative', 'NOT negative']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cm = ax.matshow(matBinary)
    plt.title("Confusion matrix - Tweets arranged by sentiment", y=1.2)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                rotation_mode="anchor")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(len(matBinary)):
        for j in range(len(matBinary)):
            text = ax.text(j, i, matBinary[i, j],
                        ha="center", va="center", color="w")
        # Create colorbar
    fig.colorbar(cm)
    plt.show()

def maxwords(column):
    phrase_len = column.apply(lambda p: len(p.split(' ')))
    max_phrase_len = phrase_len.max()
    print('max phrase len: {0}'.format(max_phrase_len))
    """plt.figure(figsize = (10, 8))
    plt.hist(phrase_len, alpha = 0.2, density = True)
    plt.xlabel('phrase len')
    plt.ylabel('probability')
    plt.grid(alpha = 0.25)
    plt.show()"""
    return  max_phrase_len

if __name__ == "__main__":
    files=["nm_nostopwords","m_nostopwords","m_stopwords","nm_stopwords","m_NLTKstopwords","nm_NLTKstopwords"]
    sets,datanv_t=transform("preprocessed")
    mnb_classify(sets,datanv_t)
    svm_classify_l(sets,datanv_t)
    svm_classify_r(sets,datanv_t)
    FFNN_classify(sets)
    k_cross_val(sets)
