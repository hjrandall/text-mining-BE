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
# nltk.download('vader_lexicon')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('averaged_perceptron_tagger')

class Text_Mining():
    def  __init__(self):
        self.text = None
        self.special_char = [",",".","<",'>',"/","?",":",";","\"","{","}","[","]","*","!","@","#","$","%","^","&","(",")","_","-","=","+"]
        self.stopwords = stopwords.words("english")
        self.tokenized_sent = []
        self.filtered_words = []
        self.sentiment = {}
        self.frequency = {}

    def set_text(self,text):
        self.text = text
        return

    def get_text(self):
        return self.text
    
    def set_tokenized_sent(self):
        self.tokenized_sent = sent_tokenize(self.text)
        return

    def get_tokenized_sent(self):
        return self.tokenized_sent
    
    def set_filtered_words(self):
        tokenized_word_text = TweetTokenizer().tokenize(self.text)
        filtered_word_text = []
        for word in tokenized_word_text:
            if word not in self.stop_words and word not in self.special_char:
                    filtered_word_text.append(word.lower())
        lem = WordNetLemmatizer()
        for word in filtered_word_text:
            self.filtered_words.append(lem.lemmatize(word))
        return

    def get_filtered_words(self):
        return self.filtered_words
    
    def set_sentiment(self):
        self.sentiment = SentimentIntensityAnalyzer().polarity_scores(self.text)
        return

    def get_sentiment(self):
        return self.sentiment
    
    def set_frequency(self):
        self.frequency = FreqDist(self.filtered_words)
        return

    def get_frequency(self,num_of_words):
        return self.frequency.most_common(num_of_words)
