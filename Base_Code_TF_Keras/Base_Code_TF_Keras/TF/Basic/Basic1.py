import tensorflow as tf

x = [[1.,2.,3.], [4.,5.,6.]]

X = tf.placeholder(tf.float32 , [None , 3])


W = tf.Variable(tf.random_normal([3,2]))
b = tf.Variable(tf.random_normal([2,1]))

w_init_data = [[0.1,0.1],[0.3,0.4],[0.1,0.4]]
W_initialized = tf.Variable( w_init_data )

res = tf.matmul(X,W) + b


# runner
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print('-- x --')
print(sess.run(X , feed_dict = {X : x}))
print('-- w --')
print(sess.run(W))
print('-- b --')
print(sess.run(b))
print('-- res --')
print(sess.run(res , feed_dict = { X : x}))




