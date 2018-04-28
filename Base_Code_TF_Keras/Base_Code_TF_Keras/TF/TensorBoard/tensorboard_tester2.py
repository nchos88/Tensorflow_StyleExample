import tensorflow as tf
import numpy as np

#data
data= np.loadtxt(r'TF\data\data.csv' , delimiter = ',' , unpack = False , dtype = 'float32')
x_data = data[:,:2]
y_data = data[:,2:]

#model shape
nb_input = 2
nb_h1 = 10
nb_h2 = 20
nb_output = 3



gStep = tf.Variable(0,trainable = False , name = 'global_step')

X = tf.placeholder( tf.float32 )
Y = tf.placeholder( tf.float32 )
dict_input = {X: x_data , Y : y_data}


W1 = tf.Variable(tf.random_uniform([nb_input , nb_h1] , -1. , 1.))
b1 = tf.Variable( tf.random_uniform([nb_h1] , -1. , 1.))
L1out =  tf.nn.relu( tf.matmul( X , W1 ) + b1 )

W2 = tf.Variable( tf.random_uniform( [nb_h1 , nb_h2] , -1. , 1. ) )
b2 = tf.Variable( tf.random_uniform( [nb_h2] , -1. , 1. ))
L2out = tf.nn.relu( tf.matmul(L1out , W2) + b2 )

W3 = tf.Variable( tf.random_uniform( [nb_h2 , nb_output] , -1. , 1. ) )
b3 = tf.Variable( tf.random_uniform( [nb_output]  , -1. , 1. ))
output = tf.matmul(L2out , W3)

loss = tf.reduce_mean( - tf.reduce_sum( Y * tf.log(output) , axis = 1 ) )
op_train = tf.train.AdamOptimizer( 0.2 ).minimize(loss , global_step = gStep )

with tf.Session() as sess:
    saver = tf.train.Saver( tf.global_variables() )
    ckpt = tf.train.get_checkpoint_state(r'Tf\TensorBoard\model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess , ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())


    for step in range(2):
        sess.run( op_train , feed_dict = dict_input)
        curr_step , loss_val = sess.run( [gStep , loss] , feed_dict = dict_input  )
        print("Step :{0} , loss : {1}".format(curr_step , loss_val))
        
        saver.save(sess , r'Tf\TensorBoard\modetb.ckpt' , global_step = curr_step)
                 
        pred , trg = sess.run([output , Y] , feed_dict = dict_input)

        hits = tf.equal( tf.argmax(output, 1) , tf.argmax(Y,1) )
        acc = tf.reduce_mean( tf.cast( hits , tf.float32 ) )

    curr_acc = sess.run(acc , feed_dict = dict_input) 
    print("Acc : {0}".format(curr_acc))











