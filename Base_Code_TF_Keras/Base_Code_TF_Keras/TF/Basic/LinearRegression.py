import tensorflow as tf

x = [1,2,3]
y = [1,2,3]

X = tf.placeholder( tf.float32 , name = 'x' )
Y = tf.placeholder( tf.float32 , name = 'y' )

W = tf.Variable( tf.random_uniform([1] , -1.0 , 1.0 ) )
b = tf.Variable( tf.random_uniform([1] , -1.0 , 1.0 ))

res = W * X  + b



cost = tf.reduce_mean( tf.square( Y - res ) ) 

optimizer = tf.train.GradientDescentOptimizer( 0.02 )

op_Train = optimizer.minimize(cost)


with tf.Session() as sess :
    sess.run( tf.global_variables_initializer())
    
    for i in range(10):
        _ , loss = sess.run( [ op_Train , cost ] , feed_dict = { X : x , Y : y}  )
        _ , loss = sess.run( [ op_Train , cost ] , feed_dict = { X : x , Y : y}  )

        print("-- current W-- ")
        print(sess.run(W))
        print("-- current b --")
        print(sess.run(b))
        print(" -- current loss --")
        print( loss )
     




