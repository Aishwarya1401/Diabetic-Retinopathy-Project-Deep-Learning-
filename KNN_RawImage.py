
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pylab
from PIL import Image
import os, sys
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd


path = "gray2"

Files = []
Xlist = []
labels =[]
names = []

imlist = os.listdir(path)  
#print imlist[0]
imlist = imlist[1:]
im = np.array(Image.open(path+'/'+imlist[0])) #open one image to get the size
m,n = im.shape[0:2]


df_label = pd.DataFrame.from_csv('trainLabels.csv' )

def get_labels(i):
    l = i.split(os.path.sep)[-1].split(".")[0]
    st1 =  df_label.loc[[l]]
    to_int = int(st1.values)
    return to_int



def imagevector_label():
    for file in os.listdir(path):
        Files.append(file)
    File = Files[1:]
    immatrix = np.array([np.array(Image.open(path+'/'+file)).flatten() for file in File],'f')    
    for file in File:   
        names.append(file)
        labels.append(get_labels(file))
    
    return immatrix, np.array(labels)        




img_vect, lab = imagevector_label()


from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split


(trainRI, testRI, trainRL, testRL) = train_test_split(
	img_vect, labels, test_size=0.25, random_state=42)


for k in [1, 3, 5, 10, 20, 50,100 ]:
    print("evaluating raw pixel accuracy...")
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(trainRI, trainRL)
    acc = model.score(testRI, testRL)
    print("raw pixel accuracy: {:.2f}%".format(acc * 100))
    

