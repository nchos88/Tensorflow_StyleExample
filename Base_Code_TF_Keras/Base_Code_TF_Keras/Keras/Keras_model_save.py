import keras 
import Loader_mnist as ld
from keras.layers import Input , Conv2D , Dropout , BatchNormalization , Dense , MaxPool2D , Flatten
from keras import losses , optimizers , models
from functools import *
import numpy as np
import yaml

path = r"data\mnist\train.csv"
x , y  = ld.Loadmnist(path)
y = np.eye(10)[y] 

# model 

input = Input( (28,28,1) )
conv1 = Conv2D( 64 , (3,3) , padding = "same" , activation = "relu")
maxpool1 = MaxPool2D( (2,2) )
flat  = Flatten() 
h0 = Dense( 128 , activation = "relu")
h1 = Dense(64 , activation = "relu" )
output = Dense(10 , activation = "softmax" )

layer_squence = [ input , conv1 , maxpool1 , flat , h0 , h1 , output]
model_test = reduce( lambda f,s : s(f) , layer_squence )

model_test = models.Model( input , model_test )
model_test.compile( loss = losses.categorical_crossentropy , optimizer = optimizers.Adam() , metrics = ['acc'] )

hist = model_test.fit(x,y, batch_size = 64 , epochs = 1 , validation_split = 0.2 )


# model to yaml - this method is for only save model structure
model_yaml = model_test.to_yaml()
with open("model1.yaml" , 'w') as f:
    yaml.dump(model_yaml, f, default_flow_style=False)


# model to h5 - this method is for save both model structure and weight
model_test.save("weight.h5")

model1 = keras.models.model_from_yaml(model_yaml)

model1.compile( loss = losses.categorical_crossentropy , optimizer = optimizers.Adam() , metrics = ['acc'])

print("model1 from yaml")
hist = model1.fit(x,y, batch_size = 64 , epochs = 1 , validation_split = 0.2 )
    
print("done")








