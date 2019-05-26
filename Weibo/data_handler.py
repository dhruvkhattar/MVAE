import pdb
import pickle as pkl
import numpy as np
import re
from gensim.models import Word2Vec
import random
from tokenizer import tokenize


def load_data(phase):

    sequence_len = 100
    image_embed_input = []
    text_input = []
    text_embed_input = []
    text_embed_norm_input = []
    output = []
    real = 0
    fake = 0

    embedding_matrix = np.load('../../data/Weibo/embedding_matrix.npy')
    embedding_matrix_norm = np.load('../../data/Weibo/embedding_matrix_norm.npy')
    word_index = pkl.load(open('../../data/Weibo/word_index.pkl'))
    img_embed = pkl.load(open('../../data/Weibo/image_embed.pkl'))
    rumor_tweets_content = open('../../../WeiboRumorSet/tweets/'+phase+'_rumor_content_segmented.txt').readlines()
    nonrumor_tweets_content = open('../../../WeiboRumorSet/tweets/'+phase+'_nonrumor_content_segmented.txt').readlines()
    rumor_tweets = open('../../../WeiboRumorSet/tweets/'+phase+'_rumor.txt').readlines()
    nonrumor_tweets = open('../../../WeiboRumorSet/tweets/'+phase+'_nonrumor.txt').readlines()

    n_lines = len(rumor_tweets)
    print len(rumor_tweets_content)

    for idx in range(2, n_lines, 3):
        if rumor_tweets[idx].strip():
            ct += 1
            images = rumor_tweets[idx-1].split('|')
            for image in images:
                img = image.split('/')[-1].split('.')[0]
                if img in img_embed:
                    text = []
                    text_embed = []
                    text_embed_norm = []
                    image_embed_input.append(img_embed[img])
                    output.append([0, 1]) 
                    words = tokenize(rumor_tweets_content[ct])
                    for word in words[:sequence_len]:
                        if word in word_index:
                            text.append(word_index[word])
                        else:
                            r = random.choice(word_index.values())
                            text.append(r)
                    text_embed.append(embedding_matrix[text[-1]])
                    text_embed_norm.append(embedding_matrix_norm[text[-1]])
                    while len(text) < sequence_len:
                        text.append(0)
                        text_embed.append(np.zeros(embedding_matrix.shape[1]))
                        text_embed_norm.append(np.zeros(embedding_matrix_norm.shape[1]))
                    text_input.append(text)
                    text_embed_input.append(text_embed)
                    text_embed_norm_input.append(text_embed_norm)
                    fake += 1
                    break
    
    n_lines = len(nonrumor_tweets)
    print len(nonrumor_tweets_content)
    
    ct = -1
    for idx in range(2, n_lines, 3):
        if nonrumor_tweets[idx].strip():
            ct += 1
            images = nonrumor_tweets[idx-1].split('|')
            for image in images:
                img = image.split('/')[-1].split('.')[0]
                if img in img_embed:
                    text = []
                    text_embed = []
                    text_embed_norm = []
                    image_embed_input.append(img_embed[img])
                    output.append([1, 0])
                    words = tokenize(nonrumor_tweets_content[ct])
                    for word in words[:sequence_len]:
                        if word in word_index:
                            text.append(word_index[word])
                        else:
                            r = random.choice(word_index.values())
                            text.append(r)
                    text_embed.append(embedding_matrix[text[-1]])
                    text_embed_norm.append(embedding_matrix_norm[text[-1]])
                    while len(text) < sequence_len:
                        text.append(0)
                        text_embed.append(np.zeros(embedding_matrix.shape[1]))
                        text_embed_norm.append(np.zeros(embedding_matrix_norm.shape[1]))
                    text_input.append(text)
                    text_embed_input.append(text_embed)
                    text_embed_norm_input.append(text_embed_norm)
                    real += 1
                    break
    
    text_input = np.array(text_input)
    text_embed_input = np.array(text_embed_input)
    text_embed_norm_input = np.array(text_embed_norm_input)
    image_embed_input = np.array(image_embed_input)
    output = np.array(output)

    idx = range(output.shape[0])
    random.shuffle(idx)
    
    output = output[idx]
    text_input = text_input[idx]
    text_embed_input = text_embed_input[idx]
    text_embed_norm_input = text_embed_norm_input[idx]
    image_embed_input = image_embed_input[idx]

    print phase, real+fake
    print "Real:", real, "Fake:", fake
    np.save('../../data/Weibo/' + phase + '_text', text_input)
    np.save('../../data/Weibo/' + phase + '_text_embed', text_embed_input)
    np.save('../../data/Weibo/' + phase + '_text_embed_norm', text_embed_norm_input)
    np.save('../../data/Weibo/' + phase + '_image_embed', image_embed_input)
    np.save('../../data/Weibo/' + phase + '_label', output)


if __name__ == '__main__':
    load_data('train')
    load_data('test')
