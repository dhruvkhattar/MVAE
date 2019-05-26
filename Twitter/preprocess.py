from gensim.models import Word2Vec
import pdb
import pickle as pkl
import numpy as np
import re
import nltk
from gensim.parsing.preprocessing import STOPWORDS
from string import punctuation
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from langdetect import detect


def process_text(string):
    string = string.decode('utf-8').lower()
    string = re.sub(u"\u2019|\u2018", "\'", string)
    string = re.sub(u"\u201c|\u201d", "\"", string)
    string = re.sub(u"\u2014", "-", string)
    string = re.sub(r"http:\ ", "http:", string)
    string = re.sub(r"http[s]?:[^\ ]+", " url ", string)
    string = re.sub(r"\"", " ", string)
    string = re.sub(r"\\n", " ", string)
    string = re.sub(r"\\", " ", string)
    string = re.sub(r"[\(\)\[\]\{\}]", r" ", string)
    string = re.sub(u'['
    u'\U0001F300-\U0001F64F'
    u'\U0001F680-\U0001F6FF'
    u'\u2600-\u26FF\u2700-\u27BF]+',
    r" ", string)
    return string.split()


def get_tweets():

    ct = 0
    train_tweets = open('./data/train_posts_with_img.txt').readlines()
    text = []
    for tweet in train_tweets:
        label = tweet.split('\t')[-1].strip()
        raw_tweet = tweet.split('\t')[1]
        tweet = process_text(raw_tweet)
        tweet = ' '.join(tweet) + '\n'
        if detect(tweet) != 'en':
            ct += 1
        tweet =  tweet.encode('utf-8')
        text.append(tweet)

    print "Tweets not in english:", ct
    f = open('./data/train_tweets.txt', 'w')
    f.writelines(text)
    f.close()
    
    ct = 0
    test_tweets = open('./data/test_posts_with_img.txt').readlines()
    text = []
    for tweet in test_tweets:
        label = tweet.split('\t')[-1].strip()
        raw_tweet = tweet.split('\t')[1]
        tweet = process_text(raw_tweet)
        tweet = ' '.join(tweet) + '\n'
        if detect(tweet) != 'en':
            ct += 1
        tweet =  tweet.encode('utf-8')
        text.append(tweet)
    
    print "Tweets not in english:", ct
    f = open('./data/test_tweets.txt', 'w')
    f.writelines(text)
    f.close()

if __name__ == '__main__':
    get_tweets()
