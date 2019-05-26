import pdb
import pickle as pkl
import numpy as np
import re
from gensim.models import Word2Vec
import random
from tokenizer import tokenize


def load_data(phase):

    sequence_len = 20
    text_input = []
    image_input = []
    image_embed_input = []
    text_embed_input = []
    text_embed_norm_input = []
    output = []
    indices = []
    real = 0
    fake = 0

    embedding_matrix = np.load('../data/embedding_matrix.npy')
    embedding_matrix_norm = np.load('../data/embedding_matrix_norm.npy')
    word_index = pkl.load(open('../data/word_index.pkl'))
    img_embed = pkl.load(open('../data/image_embed.pkl'))
    img_dict = pkl.load(open('../data/'+phase+'_img.pkl'))
    tweets = open('../data/'+phase+'_posts_with_img.txt').readlines()
    translated_tweets = pkl.load(open('../data/translated_'+phase+'_tweets.pkl'))

    ct = 0
    for idx, tweet in enumerate(tweets):
        tweet = tweet.split('\t')
        tweet_id = str(tweet[0])
        tweet_text = tweet[1]
        if ct < len(translated_tweets):
                if translated_tweets[ct][0] == idx:
                    tweet_text = translated_tweets[ct][2]
                    ct += 1
            words = tokenize(tweet_text)
            if len(words) <= 2:
                continue
            text = []
            text_embed = []
            text_embed_norm = []
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

            if tweet[-1].strip() == 'fake':
                label = [0, 1]
            else:
                label = [1, 0]
          
            if phase == 'train':
                images = tweet[3].split(',')
            else:
                images = tweet[4].split(',')
           
            for image in images:
                image = image.strip()
                if image in img_dict:
                    text_input.append(text)
                    text_embed_input.append(text_embed)
                    text_embed_norm_input.append(text_embed_norm)
                    image_input.append(img_dict[image])
                    image_embed_input.append(img_embed[image])
                    output.append(label)
                    indices.append(idx)
                    if label[0] == 1:
                        real += 1
                    else:
                        fake += 1
                    break
    
    text_input = np.array(text_input)
    image_input = np.array(image_input).astype('float32') / 255.
    text_embed_input = np.array(text_embed_input)
    text_embed_norm_input = np.array(text_embed_norm_input)
    image_embed_input = np.array(image_embed_input)
    output = np.array(output)
    indices = np.array(indices)

    idx = range(output.shape[0])
    random.shuffle(idx)
    
    output = output[idx]
    text_input = text_input[idx]
    image_input = image_input[idx]
    text_embed_input = text_embed_input[idx]
    text_embed_norm_input = text_embed_norm_input[idx]
    image_embed_input = image_embed_input[idx]
    indices = indices[idx]

    print phase, real+fake
    print "Real:", real, "Fake:", fake
    np.save('../data/' + phase + '_text', text_input)
    np.save('../data/' + phase + '_image', image_input)
    np.save('../data/' + phase + '_text_embed', text_embed_input)
    np.save('../data/' + phase + '_text_embed_norm', text_embed_norm_input)
    np.save('../data/' + phase + '_image_embed', image_embed_input)
    np.save('../data/' + phase + '_social', social_input)
    np.save('../data/' + phase + '_label', output)
    np.save('../data/' + phase + '_indices', indices)


if __name__ == '__main__':
    #load_data('train')
    load_data('test')
