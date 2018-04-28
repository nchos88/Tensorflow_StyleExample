from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Model 
from keras.layers import *
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

import win_unicode_console
win_unicode_console.enable()

img_width, img_height = 256, 256
n_classes = 10
train_data_dir = "data/train"
validation_data_dir = "data/val"
nb_train_samples = 4125
nb_validation_samples = 466 
batch_size = 16
epochs = 50

model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
model.summary()

for layer in model.layers[:5]:
    layer.trainable = False

each_output = {}

for ly in model.layers:
    each_output[ly.name] = ly
    print(ly.name)

eachlayerLast = []
for i in range(1,5):
    print(i)
    string = "block{}_pool".format(i)
    eachlayerLast.append(each_output[string])

layer_nect = eachlayerLast[-1]

print(type(layer_nect))

layer_mid = K.function( [ model.layers[0].input ] , [layer_nect])


layer_output = layer_mid([x])[0]

#golabel 
batch = BatchNormalization()
upsam = UpSampling2D( (2,2) )

#specify
deconv1_4 = Conv2D(512 , (3,3) , padding = 'same' , activation = 'relu')
deconv1_3 = Conv2D(256 , (3,3) , padding = 'same' , activation = 'relu')
deconv1_2 = Conv2D(128 , (3,3) , padding = 'same' , activation = 'relu')
deconv1_1 = Conv2D(54  , (3,3) , padding = 'same' , activation = 'relu')

shortcutlist = eachlayerLast[1:-1]
deconvlist = [deconv1_2,deconv1_3,deconv1_4]

first = deconv1_1(layer_output)

## link deconvnet
def singleBlock( input , shortcut , conv ):
    uped = UpSampling2D( (2,2) )(input)
    concated = concatenate( [uped , shortcut] , axis = 1 )
    conved = conv(concated)
    return BatchNormalization()(conved)

def CreateDeconv( input , dconvs , shorcuts , n):
    if n == 1:
        return singleBlock( input , shorcuts[ -n ] , dconvs[ n ] )
    return CreateDeconv( singleBlock( input , shorcuts[ -n ] , dconvs[ n ] ) , n-1 )

unet = CreateDeconv(first , deconvlist , shortcutlist )

outputL = Conv2D( n_classes , (3,3) , padding ='same')(unet) # current shape is 

temp = outputL.output_shape




print()