import keras 
import Loader_mnist as ld
from keras import models
import numpy as np
import yaml

path = r"data\mnist\train.csv"
x , y  = ld.Loadmnist(path)
y = np.eye(10)[y] 
  

model_yaml_str = None
with open("model1.yaml" , "r") as f:
    model_yaml_str = f.read()

model1 = keras.models.model_from_yaml(model_yaml_str)
model2 = keras.models.load_model("weight.h5" )

print("model1 from yaml")
hist = model1.fit(x,y, batch_size = 64 , epochs = 1 , validation_split = 0.2 )
print("model2 from h5")
hist = model2.fit(x,y, batch_size = 64 , epochs = 1 , validation_split = 0.2 )





