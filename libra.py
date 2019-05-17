#build GUI for text file selection
import PySimpleGUI as sg      
window_rows = [[sg.Text('Please select a .txt file for analysis')],      
                 [sg.InputText(), sg.FileBrowse()],      
                 [sg.Submit(), sg.Cancel()]]      
window = sg.Window('Libra', window_rows)    
event, values = window.Read()    
window.Close()
source_filename = values[0]    

#Open selected text file and tokenize
import nltk 
from nltk import word_tokenize
f = open(source_filename, encoding = 'ISO-8859-1')
raw = f.read()
tokens = nltk.word_tokenize(raw)
tokens = nltk.wordpunct_tokenize(raw)

#remove stopwords, spaces, and punctuation
from nltk.corpus import stopwords
stoplist = set(stopwords.words("english"))
tokens = [w for w in tokens if not w in stoplist] 
for w in tokens:
    if w not in stoplist:
        pass
import string
table = str.maketrans ('', '', string.punctuation)
words = [w.translate(table) for w in tokens]
words = [word for word in tokens if word.isalpha()]

#perform a frequency distribution of top n words with above filters
from nltk import FreqDist
fdist = nltk.FreqDist(words)
fdist.plot(10) #change the n value here to whatever range you want

#bigram collocation engine
from nltk.collocations import BigramCollocationFinder
def generate_bigrams(tokens):
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(tokens, window_size = 3)
    finder.apply_word_filter(lambda w: len(w) < 3 or w.lower() in stoplist)
    finder.apply_freq_filter(1)
    colls = finder.nbest(bigram_measures.likelihood_ratio, 10)
    return colls 

#trigram collocation engine
from nltk.collocations import TrigramCollocationFinder
def generate_trigrams(tokens):
    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    finder = TrigramCollocationFinder.from_words(tokens, window_size = 3)
    finder.apply_word_filter(lambda w: len(w) < 3 or w.lower() in stoplist)
    finder.apply_freq_filter(1)
    colls = finder.nbest(trigram_measures.likelihood_ratio, 10)
    return colls 

#sentiment analysis engine
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score
sid = SentimentIntensityAnalyzer()
summary = {"positive":0,"neutral":0,"negative":0}
for x in words: 
    ss = sid.polarity_scores(x)
    if ss["compound"] == 0.0: 
        summary["neutral"] +=1
    elif ss["compound"] > 0.0:
        summary["positive"] +=1
    else:
        summary["negative"] +=1

#printed outputs: sentiment analysis, frequency distribution, and collocations
print('Sentiment Analysis:',summary)
print('_________________________________________________________________')
print('Bigrams:', generate_bigrams(words))
print('_________________________________________________________________')
print('Trigrams:', generate_trigrams(words))
print('_________________________________________________________________')
print('Performing topic modeling...')

##BEGIN LDA TOPIC MODELING PARTY##

#load spaCy and use its tokenizer for LDA
import spacy
spacy.load('en')
from spacy.lang.en import English
parser = English()
def tokenize(raw):
    lda_tokens = []
    tokens = parser(raw)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

#lemmatize words for LDA 
from nltk.corpus import wordnet as wn
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma    
from nltk.stem.wordnet import WordNetLemmatizer
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

#set tokens to be modeled with LDA
def prepare_text_for_lda(tokens):
    tokens = tokenize(raw)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in stoplist]
    tokens = [get_lemma(token) for token in tokens]
    return tokens

#output tokens into a data bank accessible by gensim
import random
text_data = []
with open(source_filename, encoding = 'ISO-8859-1' ) as g:
    for line in g:
        tokens = prepare_text_for_lda(line)
        if random.random() > .01:
            text_data.append(tokens)

#build corpus and dictionary to pass to gensim 
from gensim import corpora
dictionary = corpora.Dictionary(text_data) 
corpus = [dictionary.doc2bow(text) for text in text_data]
import pickle
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')

#iterate gensim 50 passes on target document; more passes = higher accuracy, slower processing
import gensim
NUM_TOPICS = 5 #change this value for higher topic counts; shorter texts will cluster with values >5
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=50)
ldamodel.save('model5.gensim')
topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)
    
#locally visualize intertopic model
dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
corpus = pickle.load(open('corpus.pkl', 'rb'))
lda = gensim.models.ldamodel.LdaModel.load('model5.gensim')
import pyLDAvis.gensim
lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
pyLDAvis.show(lda_display)
