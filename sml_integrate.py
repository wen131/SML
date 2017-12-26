# Wen 2017.11.22
# SML using corel5k(https//github.com/watersink/Corel5K)
# use sklearn scipy PIL numpy package

import scipy.io as sci
import scipy as sc
from scipy.stats import multivariate_normal as mnormal
import code
import numpy
import pickle
import multiprocessing

label_train=sci.loadmat("corel5k_train_annot.mat")["annot1"]

with open("corel5k_train_list.txt") as f:
    train_list=f.read().splitlines()

def ans_one_model(index,mweights,mmeans,mcovs):
    #init
    nummods=mweights.shape[0]
    initgaus=mweights.shape[1]
    finalgaus=64
    lenvars=mmeans.shape[2]
    weights=numpy.zeros(finalgaus,dtype="float128")
    means=numpy.zeros((finalgaus,lenvars),dtype="float128")
    covs=numpy.zeros((finalgaus,lenvars,lenvars),dtype="float128")
    h=numpy.float128(numpy.random.randint(1,10000,(finalgaus,nummods,initgaus)))
    h=h/h.sum(0)

    for _ in range(80):
    #M-step
        weights=h.sum(1).sum(1)/(nummods*initgaus)
        temp=h*mweights
        temp=temp/temp.sum(1).sum(1).reshape(finalgaus,1,1)
        tempcovs=(mmeans.reshape(1,nummods,initgaus,lenvars,1)-means.reshape(finalgaus,1,1,lenvars,1)).reshape(finalgaus*nummods*initgaus,lenvars,1)
        tempcovs=numpy.array([tt.dot(tt.transpose()) for tt in tempcovs]).reshape(finalgaus,nummods,initgaus,lenvars,lenvars)
        tempcovs=tempcovs+mcovs
        covs=(temp.reshape(finalgaus,nummods,initgaus,1,1)*tempcovs).sum(1).sum(1)
        means=(temp.reshape(finalgaus,nummods,initgaus,1)*mmeans.reshape(1,nummods,initgaus,lenvars)).sum(1).sum(1)
    #E-step
        for m in range(finalgaus):
            gaus=mnormal(means[m],covs[m])
            invcovs=numpy.linalg.inv(numpy.float64(covs[m]))
            temp=numpy.zeros((nummods,initgaus),dtype="float128")
            for j in range(nummods):
                for k in range(initgaus):
                    temp[j,k]=-0.5*invcovs.dot(mcovs[j][k]).trace()
            h[m,:,:]=sc.power(gaus.pdf(mmeans)*sc.power(sc.e,temp),mweights)*weights[m]+1e-20
        h=h/h.sum(0)

    with open("final_model/%d.pickle"%(index,),"wb") as f:
        pickle.dump((weights,means,covs),f)

def worker(start,end):
    print("process start to ans %d to %d"%(start,end)) 
    for i in range(start,end,-1):
        try:
            with open("final_model/%d.pickle"%(i,)) as f:
                print("final_model/%d.pickle already exit"%(i,))
        except:
            print("final_model/%d.pickle not find"%(i,))
            weights=[]
            means=[]
            covs=[]
            for j in range(label_train.shape[0]):
                if label_train[j,i]:
                    temp=models[j]
                    weights.append(temp[0])
                    means.append(temp[1])
                    covs.append(temp[2])
            weights=numpy.array(weights)
            means=numpy.array(means)
            covs=numpy.array(covs)
            print("start to ans %d with %d models"%(i,weights.shape[0]))
            try:
                ans_one_model(i,weights,means,covs)
                print("%d is done"%(i,))
            except:
                print("except in %d"%(i,))

models=[]
if __name__=="__main__":
    for index,_ in enumerate(train_list):
        with open("models/%d.pickle"%(index,),"rb") as f:
            models.append(pickle.load(f))
    print(label_train.shape)
    print(label_train.sum(0))
    unfinish=[]
    for i in range(label_train.shape[1]):
        try:
            with open("final_model/%d.pickle"%(i,)) as f:
                pass
        except:
            unfinish.append(i)
    gap=len(unfinish)//2
    ps=[]
    print(unfinish)
    ps.append(multiprocessing.Process(target=worker,args=(14,7)))
    for p in ps:
        p.start()
