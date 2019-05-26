from gensim.models import Word2Vec
import pdb
import pickle as pkl
import numpy as np
import re
from tokenizer import tokenize


def get_embeds():

    train_tweets = open('../data/train_posts_with_img.txt').readlines()
    translated_train_tweets = pkl.load(open('../data/translated_train_tweets.pkl'))
    
    tweets = []
    ct = 0
    for idx, tweet in enumerate(train_tweets):
        tweet = tweet.split('\t')[1]
        if ct < len(translated_train_tweets):
            if translated_train_tweets[ct][0] == idx:
                tweet = translated_train_tweets[ct][2]
                ct += 1
        tweet_words = tokenize(tweet)
        if len(tweet_words) > 2:
            tweets.append(tweet_words)

    print 'Max sentence length:', max([len(tweet) for tweet in tweets])
    print 'Avg sentence length:', sum([len(tweet) for tweet in tweets]) / len(tweets)
    print 'Min sentence length:', min([len(tweet) for tweet in tweets])
   
    model = Word2Vec(tweets, min_count=1, size=32)
    words = list(model.wv.vocab)
   
    word_index = {}
    ct = 1
    embedding_matrix = []
    embedding_matrix.append(np.zeros(32))
    for word in words:
        word_index[word] = ct
        ct += 1
        embedding_matrix.append(model[word])

    embedding_matrix = np.array(embedding_matrix)
   
    print "Vocab Size:", len(word_index)
    pkl.dump(word_index, open('../data/word_index.pkl', 'w'))
    model.save('../data/word_embed.bin')
    np.save('../data/embedding_matrix', np.array(embedding_matrix))
    
    max_val = np.max(embedding_matrix)
    min_val = np.min(embedding_matrix)
    embedding_matrix = (embedding_matrix - min_val) / (max_val - min_val)
    for i in range(32):
        embedding_matrix[0][i] = 0

    np.save('../data/embedding_matrix_norm', np.array(embedding_matrix))


if __name__ == '__main__':
    get_embeds()
