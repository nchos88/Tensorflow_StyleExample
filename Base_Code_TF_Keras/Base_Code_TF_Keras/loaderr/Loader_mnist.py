import csv 
import os 
import numpy as np
import cv2
import pandas as pd

from matplotlib import pyplot as plt


import functools
import itertools

def Loadmnist_slow(path , output_img = True):

    labels = []
    datas = []
    with open(path , 'r' ) as f:
        reader = csv.reader(f)
        for i , line in enumerate(reader):
            if i == 0:
                continue

            if i == 100 : 
                break
            labels.append(line[0])
            datas.append( list(map( lambda x :  x.split(',') ,line[1:] )) )
    
    labels = np.asarray(labels).astype(np.int8)
    datas = np.asarray(datas).astype(np.int8)
    if output_img:
       
        datas = datas.reshape((datas.shape[0],28,28,1))

    
    return datas, labels

def Loadmnist(path  , output_img = True ):

    labels = []
    datas = []

    res = pd.read_csv( path , header = 1 , sep = ','  , dtype = np.int32 )

    labels = res.iloc[ :, 0]
    datas = res.iloc[ : , 1:]
    
    labels = np.asarray(labels).astype(np.int8)
    datas = np.asarray(datas).astype(np.int8)
    print(datas.shape)
    if output_img:
        datas = datas.reshape((datas.shape[0],28,28,1))
    return datas, labels


if __name__ == "__main__":
    # test code
    trainpath = r"data\mnist\train.csv"
    x , y = LoadmnistP(trainpath)
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(x[i] , cmap='gray')
    
    plt.show()


        






