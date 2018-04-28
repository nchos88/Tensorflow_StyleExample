import keras 
import Loader_mnist as ld
from keras import models
import numpy as np
from quiver_engine import server


path = r"data\mnist\train.csv"
x , y  = ld.Loadmnist(path)
y = np.eye(10)[y] 


model2 = keras.models.load_model("weight.h5" )
print("model2 from h5")
server.launch(model = model2,temp_folder='./quiver/tmp',input_folder=r'data\mnist\Train_img',port=5000)

#hist = model2.fit(x,y, batch_size = 64 , epochs = 1 , validation_split = 0.2 )








