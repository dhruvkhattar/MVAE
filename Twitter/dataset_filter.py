import pdb
import pickle as pkl
import numpy as np

def load_data(phase):

    text_input = []
    
    if phase == 'train':
        f = open('../mediaeval2016/train_posts.txt')
    else:
        f = open('../mediaeval2016/test_posts.txt')
    
    tweets = f.readlines()
    tweets = tweets[1:]
    
    img_dict = pkl.load(open('../data/image_embed.pkl'))
   
    tweets_with_img = []
    for tweet in tweets:
        if phase == 'train':
            images = tweet.split('\t')[3].split(',')
        else:
            images = tweet.split('\t')[4].split(',')
       
        for image in images:
            image = image.strip()
            if image in img_dict:
                tweets_with_img.append(tweet)
                break

    print len(tweets), len(tweets_with_img)
    f = open('../data/'+phase+'_posts_with_img.txt', 'w')
    f.write(''.join(tweets_with_img))

if __name__ == '__main__':
    load_data('train')
    load_data('test')

