

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pylab
from PIL import Image
import os, sys
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split 
from sklearn.decomposition import PCA


path = "keras_final"

Files = []
Xlist = []
labels =[]
names = []



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

n_components = 500
img_vect, lab = imagevector_label()



pca = PCA(n_components=n_components).fit(img_vect)


X_train_pca = pca.transform(img_vect)

  

(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
	X_train_pca, lab, test_size=0.15,random_state=42)   
    
from sklearn.neighbors import KNeighborsClassifier

for k in [1, 3, 5, 10, 20, 50,100 ]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(trainFeat, trainLabels)
    acc = model.score(testFeat, testLabels)
    print(" feature accuracy: {:.2f}%".format(acc * 100))
    predictn = model.predict(testFeat)
    print(predictn)

    