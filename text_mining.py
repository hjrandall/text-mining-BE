import string
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

word_count = 25
with open("sample_text.txt") as mytext:
    # set up for the text mining
    text = mytext.read()
    # print(sumarizer)
    stop_words = set(stopwords.words("english"))
    special_char = [",",".","<",'>',"/","?",":",";","\"","{","}","[","]","*","!","@","#","$","%","^","&","(",")","_","-","=","+"]
    # tokenizing the raw text data
    tokenized_sent_text = sent_tokenize(text)
    tokenized_word_text = TweetTokenizer().tokenize(text)
    # now we need to filter the tokenized words to check for stop words and punctuation
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
    frequency = FreqDist(lem_words)
    print(frequency.most_common(word_count))
    # print(lem_words)
    # print (frequency.most_common)
    # print( tokenized_sent_text)
    # print( tokenized_word_text)