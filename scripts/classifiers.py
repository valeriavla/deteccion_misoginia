import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
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

def mnb_classify(sets,datanv): 
    X=np.concatenate((sets[0],sets[1]))
    y=np.concatenate((sets[2],sets[3]))    
    mnb = MultinomialNB()
    scores = cross_val_score(mnb, X, y, scoring='accuracy', cv=StratifiedKFold(n_splits=10), n_jobs=-1)
    print('Accuracy with cross validation: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    mnb.fit(sets[0],sets[2])
    y_pred = mnb.predict(sets[1])
    y_list=y_pred.tolist()
    with open("pred_mnb.csv", "w") as file_csv:
        file_csv.write("predicted,label,value"+"\n")
        for i in range (0,len(y_list)):
            file_csv.write(str(y_list[i])+","+str(sets[3][i])+","+str(datanv[i])+"\n")
    print("MNB Accuracy:",metrics.accuracy_score(sets[3], y_pred))
    print(metrics.classification_report(sets[3], y_pred))
    matBinary = metrics.confusion_matrix(sets[3], y_pred)
    print(confusion_matrix(sets[3], y_pred))

def svm_classify(sets,datanv):
    X=np.concatenate((sets[0],sets[1]))
    y=np.concatenate((sets[2],sets[3]))  
    model = SVC(kernel='linear')
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=StratifiedKFold(n_splits=5), n_jobs=-1)
    print('Accuracy with cross validation: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    expected = sets[2]
    model.fit(sets[0], sets[2])
    predicted = model.predict(sets[0])
    y_list=predicted.tolist()
    with open("pred_svm.csv", "w") as file_csv:
        file_csv.write("predicted,label,value"+"\n")
        for i in range (0,len(y_list)):
            file_csv.write(str(y_list[i])+","+str(sets[3][i])+","+str(datanv[i])+"\n")
    print("SVM Accuracy:",metrics.accuracy_score(sets[2], predicted))
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    

def LSTM_classify(sets):
    """
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    """
    num_words=6967
    max_phrase_len = 45
    #[X_train,X_test,y_train,y_test]
    model_lstm = Sequential()
    model_lstm.add(Embedding(input_dim = num_words, output_dim = 256, input_length = max_phrase_len))
    model_lstm.add(Masking(mask_value=0.0))
    model_lstm.add(LSTM(256, dropout = 0.3, recurrent_dropout = 0.3))
    model_lstm.add(Dense(256, activation = 'relu'))
    model_lstm.add(Dropout(0.3))
    model_lstm.add(Dense(5, activation = 'softmax'))
    model_lstm.compile(
        loss='categorical_crossentropy',
        optimizer='Adam',
        metrics=['accuracy']
    )
    history = model_lstm.fit(sets[0], sets[2], 
                    batch_size=2048, epochs=150,
                    validation_data=(sets[1], sets[3]))
    model_lstm.evaluate(sets[1], sets[3])
    """model = Sequential()
    model.add(
        Embedding(input_dim=num_words,
                input_length = max_phrase_len,
                output_dim=100,
                weights=[embedding_matrix],
                trainable=False,
                mask_zero=True))
    model.add(Masking(mask_value=0.0))
    model.add(LSTM(64, return_sequences=False, 
                dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_words, activation='softmax'))
    model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train,  y_train, 
                    batch_size=2048, epochs=150,
                    callbacks=callbacks,
                    validation_data=(X_valid, y_valid))
    model.evaluate(X_valid, y_valid)"""

def divide_dataset_random(data,labels):
    X_train, X_test, y_train, y_test = train_test_split(
           data,labels,
           test_size = 0.25, random_state = 42)
    return[X_train,X_test,y_train,y_test,data]

def divide_dataset(data,labels,data_t,labels_t):
    X_train=data
    X_test=data_t
    y_train=labels
    y_test=labels_t
    return[X_train,X_test,y_train,y_test,data]

def vectorize(column):
    vectorizer = CountVectorizer()
    vectorizer.fit(column)
    vector = vectorizer.transform(column)
    return vector.toarray()

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
    data_vect=vectorize(data)
    data_vect_t=np.concatenate((data_vect[850:1100],data_vect[-250:]))
    return divide_dataset(data_vect,labels,data_vect_t,labels_t),data_t

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
    #print(metrics.SCORERS.keys())
    files=["nm_nostopwords","m_nostopwords","m_stopwords","nm_stopwords","m_NLTKstopwords","nm_NLTKstopwords"]
    sets,datanv_t=transform("preprocessed")
    
    mnb_classify(sets,datanv_t)
    #svm_classify(sets,datanv_t)
    #LSTM_classify(sets)
