import numpy as np
import cv2
import os 
import pandas as pd


def ToImage(path,outputdir):

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    res = pd.read_csv( path ,  header = 1 , sep = ','  , dtype = np.int32 )
    labels = res.iloc[ :, 0]
    datas = res.iloc[ : , 1:]

    labels = np.asarray(labels).astype(np.int8)
    datas = np.asarray(datas * 255).astype(np.int8)
    datas = datas.reshape((datas.shape[0],28,28,1))

    for i , imgdata in enumerate(datas):
        name = str(i) + "_{}.jpg".format(labels[i]) 
        cv2.imwrite( os.path.join( outputdir, name ) , imgdata)

if __name__ == "__main__":
    outputdir  = r"data\mnist\Train_img"
    path = r"data\mnist\train.csv"
    ToImage(path , outputdir)









