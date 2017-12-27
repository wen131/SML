# Wen 2017.11.22
# SML using corel5k(https//github.com/watersink/Corel5K)
# use sklearn scipy PIL numpy package

import PIL.Image as Image
import scipy.io as sio
import scipy as sc
from scipy.fftpack import dct
from scipy.stats import multivariate_normal as mnormal
import numpy
import pickle
import multiprocessing

label_train=sio.loadmat("corel5k_train_annot.mat")["annot1"]
label_test=sio.loadmat("corel5k_test_annot.mat")["annot2"]

with open("corel5k_test_list.txt") as f:
    test_list=f.read().splitlines()


def ans_one_block(b):
    b=b.reshape(64,3).transpose()
    return dct(b)[:,:10].reshape(10*3)

# split to N blocks and return N*30 matrix
# in corel5k N is always a const
def ans_one_image(pic): 
    p=Image.open("%s.jpeg"%(pic,))
    p=p.convert("YCbCr")
    x,y=p.size
    res=[]
    color_matrix=numpy.array([[p.getpixel((j,i)) for j in range(x)] for i in range(y)])
    for i in range(0,x-7,2):
        for j in range(0,y-7,2):
            res.append(ans_one_block(color_matrix[j:j+8,i:i+8]))
    return res

class GaussMixture():
    def __init__(self,params):
        self.n=len(params[0])
        self.gauss=[mnormal(params[1][i],params[2][i]) for i in range(self.n)]
        self.weights=params[0]

    def logpdf(self,datas):
        return [sc.log10((self.weights*[self.gauss[i].pdf(d) for i in range(self.n)]).sum(0)+1e-300) for d in datas]

def worker(start,end):
    for i in range(start,end):
        print("start to process %d"%(i,))
        bags=ans_one_image(test_list[i])
        res=[sum(m.logpdf(bags))+frequence[i] for i,m in enumerate(models)]
        res=sorted(enumerate(res),key=lambda x:x[1],reverse=True)[:5]
        print("%d is done:\n"%(i,))
        print(res)
        with open("predict/%d.pickle"%(i,),"wb") as f:
            pickle.dump(res,f)

output=numpy.zeros(label_test.shape)
models=[]
frequence=[sc.log10(_) for _ in label_train.sum(0)/label_train.shape[0]]

if __name__=="__main__":
    for i in range(label_test.shape[1]):
        with open("final_model/%d.pickle"%(i,),"rb") as f:
            models.append(GaussMixture(pickle.load(f)))
    ps=[]
    gap=20
    ps=[multiprocessing.Process(target=worker,args=(i*gap,(i+1)*gap)) for i in range(15)]
    for p in ps:
        p.start()
