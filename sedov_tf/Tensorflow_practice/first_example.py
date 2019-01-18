import tensorflow as tf

# we want to implement z  wx+b in low level api tensorflow

## create a graph

g = tf.Graph()

with g.as_default():
    x = tf.placeholder(dtype=tf.float32, shape=(None), name = 'x')
    w = tf.Variable(initial_value= 2.0, name='weight')
    b = tf.Variable(initial_value=0.7, name='bias')

    z = w*x + b

    init = tf.global_variables_initializer()
## after building the graph we need to create a session and pass the graph g to it

with tf.Session(graph=g) as sess:
    ## initialize w and b:
    sess.run(init)

    ## evaluate x:

    for t in [1.0, 0.6, -1.8]:
        print('x={:4.1f} --> z={:4.1f}'.format(t,sess.run(z, feed_dict={x:t})))


# a place holder is used to represent variables that will be used to feed in the data to the graph

# varibale is used to represent training variables

# a graph initially is setup that will dictate how the computation should take place and later start the execusion
# execusion is to pump data continuously to the computation graph while tweaking the weights.