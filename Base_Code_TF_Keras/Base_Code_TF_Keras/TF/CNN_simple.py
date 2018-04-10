import tensorflow as tf 
import Loader_mnist as ld
import numpy as np
from sklearn.model_selection import train_test_split
from functools import *

def mul_shape(tenseorshape):
    return reduce( lambda f,s : f*s , tenseorshape.as_list()[1:])

datapath = r"data\mnist\train.csv"
x , y = ld.Loadmnist(datapath, True)
y = np.eye( 10 )[y.reshape(-1)]

print(x.shape)

xTrain , xTest ,yTrain , yTest = train_test_split(x , y , test_size = 0.2 )
print("data type = {}".format( type(xTrain)))

nb_b , nb_h , nb_w , _ = xTrain.shape

w2_nb = 128
output_nb = 10

## Model Build 

X = tf.placeholder( shape = ( None , nb_h , nb_w , 1), dtype = tf.float32 )
Y = tf.placeholder( shape = ( None ) ,  dtype = tf.float32)


conv1_relu = tf.layers.conv2d( inputs = X , filters = 32 , kernel_size =  (  3,3 ) , padding = "SAME" , activation = tf.nn.relu )
conv1_relu_maxpool = tf.layers.max_pooling2d( conv1_relu , [2,2] , 2 , "SAME")


temp = reduce( lambda f,s : f*s , conv1_relu_maxpool.get_shape().as_list()[1:])
flatL = tf.reshape( conv1_relu_maxpool , [-1 , mul_shape( conv1_relu_maxpool.get_shape() ) ] )

# version1 
w1_shape = ( flatL.get_shape().as_list()[1] , w2_nb ) 

W1 = tf.Variable( tf.random_normal( w1_shape ) , dtype = tf.float32 )
b1 = tf.Variable( tf.random_normal( shape = [w2_nb]  ) , dtype = tf.float32 )
h1 = tf.sigmoid( ( tf.matmul( flatL , W1 ) + b1 ) )

w2_shape = [ h1.get_shape().as_list()[1] , output_nb]

W2 = tf.Variable( tf.random_normal( w2_shape ) , dtype = tf.float32)
b1 = tf.Variable( tf.random_normal( shape = [output_nb] ) , dtype = tf.float32)

output = tf.nn.softmax( tf.matmul( h1 , W2 ) + b1 )
predict = tf.argmax( output , dimension = 1 ) 


# version2 - use layer
h1_v2 = tf.layers.dense( inputs =  flatL , units = 128 , activation = tf.sigmoid )
output_v2 = tf.layers.dense( inputs = h1_v2 , units = 10 , activation = tf.nn.softmax )
predict_v2 = tf.argmax( output_v2 , dimension = 1 ) 


# LOSS
loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( logits = output_v2, labels = Y ) )
loss_v1 = tf.reduce_mean( tf.losses.log_loss( predictions = output_v2, labels = Y ) )
#loss_v1 = tf.reduce_mean( tf.losses.cross( logits = output_v2, labels = Y ) )

#optimizer  
sgd =tf.train.GradientDescentOptimizer( 0.2 )

# minimize 
train_op = sgd.minimize( loss = loss_v1 , global_step = tf.train.get_global_step())


# Start Session 
iter = 100
print("Start")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(iter):
        _ , loss_train = sess.run([train_op ,loss ] , feed_dict = {X : xTrain , Y : yTrain })
        print("Iter : {} Loss : {}".format( i , loss_train))
        if i%10 == 0:
            loss_val = sess.run( [loss] , feed_dict = {X : xTest , Y : yTest} )
            print("Validation = {}".format(loss_val))

print( conv1_relu_maxpool.get_shape().as_list() )
print( flatL.get_shape().as_list() )








