import numpy as np
from keras.layers import *
import Loader_mnist as ld
from sklearn.model_selection import train_test_split 
from keras import models
from keras.layers import *
from keras import losses ,optimizers 
from keras import backend as K
from keras import metrics
import numpy as np

from functools import *


# setting 
iter = 200



# data
path = r"data\mnist\train.csv"
x , y = ld.Loadmnist(path, True )

y = np.eye(10)[y]


xTrain , xTest , yTrain , yTest = train_test_split( x ,y , test_size = 0.3 ) 

num_b, num_h , num_w , num_c = xTrain.shape


input = Input( (num_h , num_w , 1) )

# custom metric
def pred_overhalf(y_true, y_pred):
    print(type())
    return K.count_params( filter( lambda x : x > 0.5 , y_pred ) )

# model - funtional 
x = Conv2D( 64 , (3,3) , activation = "relu")(input)
x = MaxPool2D( padding = "same" )(x)
x = Flatten()(x)
x = Dense( 128 , activation = 'sigmoid' )(x)
output = Dense( 10 , activation = 'sigmoid')(x)

model_ver1 = models.Model( input , output ) 
model_ver1.compile( optimizer = optimizers.Adam() , loss = losses.categorical_crossentropy , metrics = [metrics.mae , metrics.categorical_accuracy])


 # model - funtional 

conv1 = Conv2D( 64 , (3,3) , activation = "relu")
maxpool1 = MaxPool2D( padding = "same" )
flat = Flatten()
d1  =Dense( 128 , activation = 'sigmoid' )
d2 = Dense( 10 , activation = 'sigmoid')

modelpipe = [input , conv1 , maxpool1 , flat , d1 , d2 ]

linked_m1 = reduce( lambda f,s : s(f) , modelpipe )

model_ver2 = models.Model( input , linked_m1 )
model_ver2.compile(optimizer = optimizers.SGD() , loss = losses.categorical_crossentropy , metrics = ['acc' ])


history = model_ver2.fit( xTrain, yTrain , batch_size = 64 , epochs = 3 , validation_split = 0.2)

score = model_ver2.evaluate( xTest , yTest )

acclist = history.history['acc']

print("done")
















