import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from itertools import chain
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
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

def mnb_classify(sets):   
    mnb = MultinomialNB()
    mnb.fit(sets[0],sets[2])
    MultinomialNB()
    y_pred = mnb.predict(sets[1])
    print("MNB Accuracy:",metrics.accuracy_score(sets[3], y_pred))
    print(metrics.classification_report(sets[3], y_pred))
    matBinary = metrics.confusion_matrix(sets[3], y_pred)
    print(confusion_matrix(sets[3], y_pred))
    plot(matBinary)

def svm_classify(sets):
    model = SVC(kernel='linear')
    model.fit(sets[0], sets[2])
    print(model)
    expected = sets[2]
    predicted = model.predict(sets[0])
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

def divide_dataset(data,labels):
    X_train, X_test, y_train, y_test = train_test_split(
           data,labels,
           test_size = 0.25, random_state = 0)
    return[X_train,X_test,y_train,y_test]

def vectorize(column):
    vectorizer = CountVectorizer()
    vectorizer.fit(column)
    vector = vectorizer.transform(column)
    return vector.toarray()

def transform(filenamem,filenamenm):
    filem=r'{0}.csv'.format(filenamem)
    filenm=r'{0}.csv'.format(filenamenm)
    data_framem=pd.read_csv(filem,encoding='latin-1')
    data_framenm=pd.read_csv(filenm,encoding='latin-1')
    misogino = data_framem["misogino"]
    no_misogino = data_framenm["no_misogino"]
    data = list(chain(misogino, no_misogino))
    labels = np.concatenate((np.ones(len(misogino)),np.zeros(len(no_misogino))))
    data_vect=vectorize(data)
    return divide_dataset(data_vect,labels)

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
    files=["m_stopwords","m_nostopwords","m_NLTKstopwords","nm_stopwords","nm_nostopwords","nm_NLTKstopwords"]
    sets=transform("m_NLTKstopwords","nm_NLTKstopwords")
    mnb_classify(sets)
    svm_classify(sets)
    #LSTM_classify(sets)