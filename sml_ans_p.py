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
import multiprocessing

label_train=sc.loadmat("corel5k_train_annot.mat")["annot1"]
label_test=sc.loadmat("corel5k_test_annot.mat")["annot2"]

mod=GaussianMixture(8)


def ans_one_block(b):
    b=b.reshape(64,3).transpose()
    return dct(b)[:,:10].reshape(10*3)

# split to N blocks and return N*30 matrix
# in corel5k N is always a const
def ans_one_image(index,pic):
    p=Image.open("%s.jpeg"%(pic,))
    p=p.convert("YCbCr")
    x,y=p.size
    res=[]
    color_matrix=numpy.array([[p.getpixel((j,i)) for j in range(x)] for i in range(y)])
    for i in range(0,x-7,2):
        for j in range(0,y-7,2):
            res.append(ans_one_block(color_matrix[j:j+8,i:i+8]))
            print(ans_one_block(color_matrix[j:j+8,i:i+8]))
    print(len(res))
    mod.fit(res)
    with open("models/%d.pickle"%(index,),"wb") as f:
        pickle.dump((mod.weights_,mod.means_,mod.covariances_),f)
    print("%d is done"%(index,))

with open("corel5k_train_list.txt") as f:
    train_list=f.read().splitlines()
with open("corel5k_test_list.txt") as f:
    test_list=f.read().splitlines()

def worker(start,end):
    print(start,end)
    for index,_ in enumerate(train_list[start:end]):
        ans_one_image(start+index,_)

if __name__=="__main__":
    gap=len(train_list)//4
    ps=[]
    for i in range(4):
        ps.append(multiprocessing.Process(target=worker,args=(gap*i,gap*(i+1))))
    for p in ps:
        pass#p.start()
    worker(0,1)
