import string
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.probability import  FreqDist
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('vader_lexicon')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('averaged_perceptron_tagger')
word_count = 25
with open("sample_text.txt") as mytext:
    text = mytext.read()
    stop_words = set(stopwords.words("english"))
    special_char = [",",".","<",'>',"/","?",":",";","\"","{","}","[","]","*","!","@","#","$","%","^","&","(",")","_","-","=","+"]
    tokenize_text = TweetTokenizer()
    tokenized_word_text = tokenize_text.tokenize(text)
    filtered_word_text =[]
    for word in tokenized_word_text:
        if word not in stop_words and word not in special_char:
                filtered_word_text.append(word.lower())
    lem = WordNetLemmatizer()
    lem_words =[]
    for word in filtered_word_text:
         lem_words.append(lem.lemmatize(word))
    sentiment = SentimentIntensityAnalyzer()
    feeling = sentiment.polarity_scores(text)
    print(feeling)
    frequency = FreqDist(lem_words)
    print(frequency.most_common)
    # print(frequency.most_common(word_count))
    # print(lem_words)
    # print (frequency.most_common)
    # print( tokenized_sent_text)
    # print( tokenized_word_text)