import pandas as pd
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
plt.style.use("fivethirtyeight")
from cycler import cycler
import pandas as pd
import string
import requests
import nltk
import spacy
from wordcloud import WordCloud, STOPWORDS
from nltk.tokenize import TweetTokenizer
from es_lemmatizer import lemmatize
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.util import suffix_replace, prefix_replace
from nltk.stem.api import StemmerI
from nltk.stem import SnowballStemmer
from sklearn.metrics import mean_squared_error,r2_score
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import nltk.collocations as collocations
from nltk import FreqDist, bigrams
import seaborn as sns

def preproces_dataset(data_frame,filename,tknzr,stopwords):
    punctuation={'"','#','$','%','&','(',')','*','+',',','-','/',':',';','<','=','>','@','[',']','^','_','`','{','|','}','~','.'}
    clean_set=[]
    set_string=""
    regex = "\\b([a-zA-Z0-9])\\1\\1+\\b"
    printable = set(string.printable)
    p = re.compile(regex, re.IGNORECASE)
    stopwords=["a","de","por","lo","le","en","y","se","e","que","con"]
    df=pd.read_csv(r'palabras_busqueda.csv')
    misspellings = df["adjetivos_errores"]
    spellings = df["adjetivos_limpios"]
    for value in data_frame:
        #Eliminar basura
        value=re.sub(r'\x93',r'', value)
        value=re.sub(r'\x94',r'', value) 
        value=re.sub(r'\x96',r'', value)
        value=re.sub(r'\x97',r'', value) 
        value=re.sub(r'\x84',r'', value)
        value=re.sub(r'\x95',r'', value)
        value=re.sub(r'\x92',r'', value)
        #separar las palabras por tokens
        tokenized_value=tknzr.tokenize(value)
        clean_tokenized=[]
        ii=False
        for word in tokenized_value:
            #separar hashtags, hashtags en minúscula = #hashtag hashtags separados por mayúsculas se transforman en palabras
            #las palabras de los hashtags deben regresar al conjunto para limpiarse
            if word[0]=="#":
                hashtag_words=re.findall('[A-Z][a-z]*',word)
                if len(hashtag_words)==0 or word[1].islower():
                    clean_tokenized.append("#hashtag")
                    continue
                else:
                    ii=tokenized_value.index(word)
                    tokenized_value.pop(ii)
                    for j in range ((len(hashtag_words)-1),-1,-1):
                        tokenized_value.insert(ii,hashtag_words[j].lower())
            if ii is False:
                    val=word
            else:
                val=tokenized_value[ii]
                ii=False
            #remover hipervinculos
            if "http" in val:
                continue
            #remover stopwords
            elif val in stopwords:
                continue
            #remover signos de puntuación
            elif val in punctuation:
                continue
            #transformar palabras con errores ortográficos
            elif val in misspellings:
                misspellings.index(val)
                clean_tokenized.append(spellings[misspellings.index(val)])
                continue
            #sustituir menciones
            elif val[0]=="@":
                clean_tokenized.append("@user")
                continue
            #eliminar caracteres continuos repetidos más de 2 veces
            elif (re.search(r'(.)\1{1,}\1{1,}', val)):
                clean_word=re.compile(r'(.)\1{1,}\1{1,}', re.IGNORECASE).sub(r'\1', val)
                if clean_word not in string.punctuation:
                    clean_tokenized.append(clean_word.lower())
                continue
            #convertir a minúsculas
            else:
                clean_tokenized.append(val.lower())
        clean_set.append(clean_tokenized)
        str1 = " " 
        str1=str1.join(clean_tokenized)+" { \n"
        with open(filename+"_NLTKstopwords.csv", "a") as preprocessed_file:
            preprocessed_file.write(str1)
        set_string+=str1
    with open(filename+"_NLTKstopwords.txt", "w") as txt_file:
        txt_file.write(set_string)
    return set_string

def create_wordcloud(set):
    comment_words = ''
    for val in set:
        comment_words += " ".join(val)+" "
    wordcloud = WordCloud(width = 800, height = 800,
                    background_color ='white',
                    min_font_size = 10).generate(comment_words)                      
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.show()

def Frequency_words_string(words,words2):
    words=words+words2
    words=words.replace(" { "," ")
    words=words.replace("\n"," ")
    words=words.replace("  "," ")
    words = words.split(' ')
    #ngrams = bigrams(words)
    all_fdist = FreqDist(words)
    #common_fdist = FreqDist(words).most_common(20)
    return all_fdist

def histogram():
    #arg all_fdist
    #all_fdist = pd.Series(dict(all_fdist))
    #pd.DataFrame(all_fdist).to_csv("frequency_bigram2.csv",encoding='latin-1')
    #common_fdist = pd.Series(dict(common_fdist))
    #colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors=['#E24A33', '#348ABD', '#988ED5','#9f419b', '#FBC15E', '#8EBA42', '#FFB5B8']
    mpl.rcParams['axes.prop_cycle'] = cycler(color=colors)
    df_rep=pd.read_csv(r'freq_nsw.csv',encoding='latin-1')
    fig, ax = plt.subplots(figsize=(10,10))
    ax.tick_params(colors='black', direction='out')
    for tick in ax.get_xticklabels():
        tick.set_color('black')
    for tick in ax.get_yticklabels():
        tick.set_color('black')
    all_plot = sns.barplot(x=df_rep["palabra"], y=df_rep["frecuencia"], ax=ax)
    plt.xticks(rotation=30)
    plt.show()

def stopwords_analyse():
    data_frame=pd.read_csv(r'frequency_nm_nsw.csv',encoding='latin-1')
    data_frame_stp=pd.read_csv(r'stopwords_cat.csv',encoding='latin-1')
    #conjunciones_bi
    #stopwords_cat=["conjunciones_uni","p_personal","p_posesivo","p_demostrativo",
                   # "p_relativos","p_indefinido","p_interrogativo","estar","ser","tener","parecer"]
    stopwords_cat=["prepos"]
    repetitions_cat={}
    print("total de palabras: ",len(data_frame["unigram"]))
    for category in stopwords_cat:
        total=0
        clean_words=[x.replace(" ","") for x in data_frame_stp[category].dropna()]
        print("stopwords are: ",clean_words)
        for i in range(0, len(data_frame["unigram"])):
            if data_frame["unigram"][i] in clean_words:
                total+=int(data_frame["num_uni"][i])
                print(data_frame["unigram"][i],data_frame["num_uni"][i])
        print(total)
        repetitions_cat[category]=total
        
    df_rep=pd.DataFrame(repetitions_cat.items(), columns=['category', 'repetitions'])
    pd.DataFrame(df_rep).to_csv("frequency_cat_prepo.csv",encoding='latin-1')
    histogram(df_rep)

def lemmatizing_freeling(filename):
    file_origin=r"{0}.txt".format(filename)
    file_lemm=r"{0}_lemm_f.csv".format(filename)
    files = {'file': open(file_origin, 'rb')}
    params = {'outf': 'tagged', 'format': 'json'}
    url = "http://www.corpus.unam.mx/servicio-freeling/analyze.php"
    r = requests.post(url, files=files, params=params)
    obj = r.json()
    lemmatized_set=[]
    lemmatized_tweet=""
    for sentence in obj:
        for word in sentence:
                lemmatized_tweet+=(word["lemma"])+" "
    lemmatized_tweet=lemmatized_tweet.replace(" { ","\n")
    with open(file_lemm, "w") as file_csv:
        file_csv.write(lemmatized_tweet)
    return lemmatized_set

def lemmatizing_spacy(tknzr,filename):
    file_origin=r"{0}.txt".format(filename)
    nlp = spacy.load('es_core_news_sm')
    f = open(file_origin, encoding="utf-8")
    text=f.read()
    f.close()
    doc = nlp(text)
    file_lemm=r"{0}_lemm_s.csv".format(filename)
    file_csv = open(file_lemm, "w")
    lemmatized_tweet=""
    for word in doc:
                lemmatized_tweet+= " "+(word.lemma_)+" "
    lemmatized_tweet=lemmatized_tweet.replace(" { "," ")
    with open(file_lemm, "w") as file_csv:
        file_csv.write(lemmatized_tweet)
    return ' '.join([word.lemma_ for word in doc])
   
def stemming(filename):
    file_origin=r"{0}.txt".format(filename)
    file_line = open(file_origin, encoding="utf-8")
    text=file_line.read()
    file_line.close()
    stemmer = SnowballStemmer("spanish")
    stemmed_tweet=""
    file_lemm=r"{0}_stemm.csv".format(filename)
    file_csv = open(file_lemm, "w")
    text=text.split(" ")
    for word in text:
                stemmed_tweet+=stemmer.stem(word)+" "
                stemmed_tweet=stemmed_tweet.replace(" { "," ")
    with open(file_lemm, "w") as file_csv:
        file_csv.write(stemmed_tweet)
    file_csv.close()

if __name__ == "__main__":
    stop_words = set(stopwords.words('spanish')) 
    tknzr = TweetTokenizer()
    data_frame=pd.read_csv(r'conjunto_sucio.csv',encoding='latin-1')
    tokens_m=preproces_dataset(data_frame["misogino"],"m",tknzr,stop_words)
    tokens_nm=preproces_dataset(data_frame["no_misogino"],"nm",tknzr,stop_words)
    create_wordcloud(tokens_m)
    create_wordcloud(tokens_nm)
    stopwords_analyse()
    lemm_f_m=lemmatizing_freeling("nm_stopwords")
    histogram()
    sp = spacy.load('es_core_news_sm')
    print(sp.Defaults.stop_words)
    lemm_f_nm=lemmatizing_freeling(tokens_nm)
    lemmatizing_spacy(tknzr,"nm_stopwords")
    files=["m_stopwords","m_nostopwords","m_NLTKstopwords","nm_stopwords","nm_nostopwords","nm_NLTKstopwords"]
    for filename in files:
        stemming(filename)