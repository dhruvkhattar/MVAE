from gensim.models import Word2Vec
import pdb
import pickle as pkl
import numpy as np
import re
from tokenizer import tokenize


def get_embeds():

    embed_dim = 32

    rumor_tweets = open('../../../WeiboRumorSet/tweets/train_rumor_content_segmented.txt').readlines()
    nonrumor_tweets = open('../../../WeiboRumorSet/tweets/train_nonrumor_content_segmented.txt').readlines()
    
    tweets = rumor_tweets + nonrumor_tweets
    tweets = [tokenize(tweet) for tweet in tweets]
    print 'Max sentence length:', max([len(tweet) for tweet in tweets])
    print 'Avg sentence length:', sum([len(tweet) for tweet in tweets]) / len(tweets)
    print 'Min sentence length:', min([len(tweet) for tweet in tweets])
   
    model = Word2Vec(tweets, min_count=1, size=embed_dim)
    words = list(model.wv.vocab)
   
    word_index = {}
    ct = 1
    embedding_matrix = []
    embedding_matrix.append(np.zeros(embed_dim))
    for word in words:
        word_index[word] = ct
        ct += 1
        embedding_matrix.append(model[word])

    embedding_matrix = np.array(embedding_matrix)
   
    print "Vocab Size:", len(word_index)
    pkl.dump(word_index, open('../../data/Weibo/word_index.pkl', 'w'))
    model.save('../../data/Weibo/word_embed.bin')
    np.save('../../data/Weibo/embedding_matrix', np.array(embedding_matrix))
    
    max_val = np.max(embedding_matrix)
    min_val = np.min(embedding_matrix)
    embedding_matrix = (embedding_matrix - min_val) / (max_val - min_val)
    for i in range(embed_dim):
        embedding_matrix[0][i] = 0

    np.save('../../data/Weibo/embedding_matrix_norm', np.array(embedding_matrix))


if __name__ == '__main__':
    get_embeds()
