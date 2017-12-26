# Wen 2017.11.22
# SML using corel5k(https//github.com/watersink/Corel5K)
# use sklearn scipy PIL numpy package

import PIL.Image as Image
import scipy.io as sc
from scipy.fftpack import dct
from sklearn.mixture import GaussianMixture
import code
import numpy
import pickle

label_train=sc.loadmat("corel5k_train_annot.mat")["annot1"]
label_test=sc.loadmat("corel5k_test_annot.mat")["annot2"]

mod=GaussianMixture(64)


def ans_one_block(b):
    b=b.reshape(64,3).transpose()
    return dct(b)[:,:10].reshape(10*3)

# split to N blocks and return N*30 matrix
# in corel5k N is always a const
def ans_one_image(p): 
    p=Image.open("%s.jpeg"%(_,))
    p=p.convert("YCbCr")
    x,y=p.size
    res=[]
    color_matrix=numpy.array([[p.getpixel((j,i)) for j in range(x)] for i in range(y)])
    for i in range(0,x-7,2):
        for j in range(0,y-7,2):
            res.append(ans_one_block(color_matrix[j:j+8,i:i+8]))
    return res

with open("corel5k_train_list.txt") as f:
    train_list=f.read().splitlines()
with open("corel5k_test_list.txt") as f:
    test_list=f.read().splitlines()

if __name__=="__main__":
    r=[]
    for index,_ in enumerate(train_list[:100]):
        print(index)
        r.extend(ans_one_image(_))
    print("start fit")
    mod.fit(r)
