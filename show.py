# Wen 2017.11.22
# SML using corel5k(https//github.com/watersink/Corel5K)
# use sklearn scipy PIL numpy package

import PIL.Image as Image
import scipy.io as sio
import random
import pickle

label_test=sio.loadmat("corel5k_test_annot.mat")["annot2"]

with open("corel5k_test_list.txt") as f:
    test_list=f.read().splitlines()
with open("corel5k_words.txt") as f:
    words=f.read().splitlines()

a=""
while(a!="q"):
    index=random.randint(0,299)
    pic=test_list[index]

    with open("predict/%d.pickle"%(index,),"rb") as f:
        model=pickle.load(f)

    p=Image.open("%s.jpeg"%(pic,))
    label=[]
    for i in range(label_test.shape[1]):
        if label_test[index,i]:
            label.append(words[i])
    predict=[words[i] for i in [m[0] for m in model]]
    print("picture %s"%(pic,))
    print("label is:\n")
    print(label)
    print("predict res is:\n")
    print(predict)
    p.show()
    a=input()
    p.close()
