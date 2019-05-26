import pickle as pkl
import pdb
import os
import numpy as np
import cv2


def photo_loader(phase):

    datadir = '../mediaeval2016/images_'+phase
    images = os.listdir(datadir)
    x = []
    x_dict = {}

    for name in images:
        fname = datadir+'/'+name
        im = cv2.imread(fname)
        try:
            im = cv2.resize(im, (232, 232), 0, 0, cv2.INTER_LINEAR)
        except:
            continue
        x.append(im)
        x_dict[name.split('.')[0]] = im

    pkl.dump(x_dict, open('../data/'+phase+'_img.pkl', 'w'))
    
    x = np.asarray(x)
    print "Images got successfully with len", x.shape
    return x

if __name__ == '__main__':
    photo_loader('train')
    photo_loader('test')
