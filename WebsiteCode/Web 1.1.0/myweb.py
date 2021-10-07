from flask import Flask, render_template, request
import pickle
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#from stop_words_list import stop_words_list
import nltk
from nltk.stem import WordNetLemmatizer
# import sklearn
# from sklearn.decomposition import LatentDirichletAllocation
# from sklearn.feature_extraction.text import CountVectorizer
import joblib

#------------------Load models by pickle-----------
with open('Data/Vectorizer_pickle', 'rb') as f:
    vectorizer = pickle.load(f)

# with open('Data/LDA_pickle', 'rb') as f:
#     lda_model = pickle.load(f)

lda_model = joblib.load('Data/lda_pickle.jl')

# lda_model = pickle.load(LatentDirichletAllocation('Data/LDA_pickle'))

#----------------- Text Cleaning --------------
# Apply text cleaning techniques
import re
import string

# nltk.download('stopwords')
# nltk.download('punkt')
stopwords = nltk.corpus.stopwords.words('english')
# print(stopwords)
from nltk import PorterStemmer
ps = PorterStemmer()

def textCleaner(text):
    ''' removing all URLs '''
    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    '''Get rid of some additional punctuation and non-sensical text that was missed the previous subroutines.'''
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    ''' Removing stop words '''
    tokens = text.split(" ")
    cleanedText = ""
    for token in tokens:
      if(token not in stopwords):
        cleanedText += (token + " ")
    text = cleanedText
    ''' Stemming '''
    tokens = text.split(" ")
    cleanedText=""
    for tok in tokens:
      cleanedText += (ps.stem(tok) + " ")
    text = cleanedText

    return text

#----------Sentiment----------------------------
from textblob import TextBlob

def mapping(input_scores):
  # f = lambda x: round((5*x/2)+3)
  f = lambda x: round((4.99*x/2)+3)

  ln = len(input_scores)
  linearMapResult = [0]*ln

  for i in range(ln):
    linearMapResult[i]= f(input_scores[i])
  return linearMapResult

#----------Sentiment with Keras ---------------------------- If not working comment this section
from keras.models import load_model

model = load_model('keras_model.h5')


def DL_predict(text, r_vote, r_verified):
    r_clean_reviews = [textCleaner(text)]
    print(r_clean_reviews)

    r_df = pd.DataFrame(r_clean_reviews, columns=['remove_lower_punct'])

    # tokenise string
    r_df['tokenise'] = r_df.apply(lambda row: nltk.word_tokenize(row[0]),
                                  axis=1)  # Row[1] is " removeLowerPunct "  column

    # remove stopwords
    r_df['remove_stopwords'] = r_df['tokenise'].apply(lambda x: [item for item in x if item not in stopwords])

    # initiate nltk lemmatiser
    wordnet_lemmatizer = WordNetLemmatizer()

    # lemmatise words
    r_df['lemmatise'] = r_df['remove_stopwords'].apply(lambda x: [wordnet_lemmatizer.lemmatize(y) for y in x])

    # join the processed data to be vectorised
    r_vectors = []

    for index, row in r_df.iterrows():
        # vectors.append(", ".join(row[6]))
        r_vectors.append(" ".join(row[3]))  # lemmatise
    # print(vectors)
    r_vectorised = vectorizer.transform(r_vectors)

    r_chi2_vectorized = chi2_selector.transform(r_vectorised)

    X_r = np.concatenate((r_chi2_vectorized.toarray(), np.array([[r_verified]]).T, np.array([[r_vote]]).T), axis=1)

    y_pred_r = model.predict(X_r, batch_size=64, verbose=1)
    y_pred_bool_r = np.argmax(y_pred_r, axis=1)

    return y_pred_bool_r[0]


#--------------Topic--------------------------
topics = ['winkler', 'purchase', 'story', 'ago', 'good movie', 'movie', 'watch year', 'dvd', 'video', 'volume']
def dom_topic(Rooz_review):
    Rooz_clean_review = Rooz_review.lower().replace("'", '').replace('[^\w\s]', ' ').replace(" \d+", " ").replace(' +',
                                                                                                                  ' ').strip()
    # print(Rooz_clean_review)

    analyser = SentimentIntensityAnalyzer()
    print('Sentiment: ', mapping([analyser.polarity_scores(Rooz_clean_review)['compound']])[0])

    # tokenise string
    Rooz_tok = nltk.word_tokenize(Rooz_clean_review)
    # print(Rooz_tok)

    # initiate stopwords from nltk
    # stop_words = stopwords.words('english')

    # add additional missing terms
    # stop_words.extend(stop_words_list)

    # remove stopwords
    Rooz1 = [item for item in Rooz_tok if item not in stopwords]
    # print(Rooz1)

    # initiate nltk lemmatiser
    wordnet_lemmatizer = WordNetLemmatizer()

    # lemmatise words
    Rooz2 = [wordnet_lemmatizer.lemmatize(y) for y in Rooz1]
    # print(Rooz2)

    # join the processed data to be vectorised

    Rooz_vectors = []

    for i in [Rooz2]:
        Rooz_vectors.append(" ".join(i))
    # print(Rooz_vectors)
    Rooz_vectorised = vectorizer.transform(Rooz_vectors)

    # print(Rooz_vectorised)

    Rooz_doc_topic = np.matrix(lda_model.transform(Rooz_vectorised))
    # print(Rooz_doc_topic)
    dominant_topic = topics[np.argmax(Rooz_doc_topic)]
    print('Dominant topic: ', dominant_topic)
    return dominant_topic


#----------------Web-------------------------

app = Flask(__name__)

@app.route('/')
def my_main_page():
    return render_template('student.html')

@app.route('/result', methods = ['POST', 'GET'])
def my_results_page():
    if request.method == 'POST':
        my_result = request.form#['command']
        my_output = 'Error. No Input'
        for key, value in my_result.items():
            my_output = {'result': [value, mapping([TextBlob(textCleaner(value)).sentiment.polarity])[0], dom_topic(value)]}
	   # my_output = {'result': [value, DL_predict(value,1,1), dom_topic(value)]} # Uncomment this one and comment the upper line for running Keras
        return render_template('result.html', result = my_output)
    #return render_template('index.html', command = None)



if __name__ == '__main__':
    app.run()
