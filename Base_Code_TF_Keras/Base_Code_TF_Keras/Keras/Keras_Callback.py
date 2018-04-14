import keras
from keras.layers import *
from keras import models , losses , optimizers
import numpy as np
import Loader_mnist as dl
from functools import *

path = r"data\mnist\train.csv"
x,y = dl.Loadmnist(path)

y = np.eye(10)[y]

input = Input( (28,28,1) )
conv1 = Conv2D( 8 , (3,3) , activation = "relu" )
maxpool1 = MaxPooling2D((2,2))
flat = Flatten()
output = Dense(10 , activation = "sigmoid" )

mlist = [input , conv1 , maxpool1 , flat , output]

lastl = reduce( lambda f,s : s(f) , mlist)

model = models.Model( input , lastl)
model.compile(loss = losses.categorical_crossentropy , optimizer = optimizers.Adam())


es =keras.callbacks.EarlyStopping( monitor = 'val_loss' , min_delta = 1.0 , patience = 2 )

outpath = r"check\weights.{epoch:02d}-{val_loss:.2f}.hdf5"
ck = keras.callbacks.ModelCheckpoint(outpath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
tb = keras.callbacks.TensorBoard(r"logs")


hist = model.fit(x,y , batch_size = 64 , epochs = 10 , validation_split = 0.2,callbacks = [es , ck , tb])







