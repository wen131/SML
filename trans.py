import pickle

for i in range(4500):
    with open("model/%d.pickle"%(i,),"rb") as f1:
        mod=pickle.load(f1)
        with open("models/%d.pickle"%(i,),"wb") as f2:
            pickle.dump((mod.weights_,mod.means_,mod.covariances_),f2)
