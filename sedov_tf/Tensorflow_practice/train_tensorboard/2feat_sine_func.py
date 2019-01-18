import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# generating the training and test dataset

W = np.arange(0.1,10,0.1)
X = np.arange(0.1,10,0.1)
data = np.ndarray(shape=[W.shape[0]*W.shape[0], 3])
counter = 0
for w in W:
    for x in X:
        y = np.sin(w*x)
        data[counter,0] = x
        data[counter,1] = w
        data[counter,2] = y

        counter = counter + 1




X_train, X_test, y_train, y_test = train_test_split(data[:,0:2],data[:,2], test_size=0.2,shuffle=False)
X_train = X_train.reshape(-1,2)
y_train = y_train.reshape(-1,1)

# defining some parameters for the neural network

learning_rate = 0.01
n_inputs = 2
n_layer1 = 10
n_layer2 = 7
n_layer3 = 4
n_out = 1

initial_std_ndist = 0.03
# generating the graph

g = tf.Graph()

with g.as_default():

    # create a place holder for the input value x
    tf_x = tf.placeholder(dtype=tf.float32, shape=[None,n_inputs], name='tf_x')
    print(tf_x)
    tf_y = tf.placeholder(dtype=tf.float32, shape=[None,n_out],name='tf_y')

    # net inputs and activation
    with tf.variable_scope(name_or_scope='layer1'):
        w1 = tf.Variable(tf.random_normal(shape=[n_inputs, n_layer1], stddev=initial_std_ndist), name='w_h1')
        tf.summary.histogram('w1',w1)
        b1 = tf.Variable(tf.random_normal(shape=[n_layer1],stddev=initial_std_ndist), name='b_h1')
        tf.summary.histogram('b1',b1)
        z_in = tf.matmul(tf_x,w1) + b1
        tf.summary.histogram('z_in',z_in)
        layer_1 = tf.nn.tanh(z_in)
        print(layer_1)
    with tf.variable_scope(name_or_scope='layer2'):
        w2 = tf.Variable(tf.random_normal(shape=[n_layer1, n_layer2], stddev=initial_std_ndist), name='w_h2')
        tf.summary.histogram('w2',w2)
        b2 = tf.Variable(tf.random_normal(shape=[n_layer2], stddev=initial_std_ndist), name='b_h2')
        tf.summary.histogram('b2',b2)
        z_h1 = tf.matmul(layer_1,w2) + b2
        tf.summary.histogram('z_h1',z_h1)
        layer_2 = tf.nn.tanh(z_h1)
        print(layer_2)

    with tf.variable_scope(name_or_scope='layer3'):
        w3 = tf.Variable(tf.random_normal(shape=[n_layer2, n_layer3], stddev=initial_std_ndist), name='w_h3')
        tf.summary.histogram('w3',w3)
        b3 = tf.Variable(tf.random_normal(shape=[n_layer3], stddev=initial_std_ndist), name='b_h3')
        tf.summary.histogram('b3',b3)
        z_h3 = tf.matmul(layer_2,w3) + b3
        tf.summary.histogram('z_h3',z_h3)
        layer_3 = tf.nn.tanh(z_h3)
        print(layer_3)
    with tf.variable_scope(name_or_scope='layer_out'):
        wout = tf.Variable(tf.random_normal(shape=[n_layer3, n_out], stddev=initial_std_ndist), name='w_out')
        tf.summary.histogram('wout',wout)
        bout = tf.Variable(tf.random_normal(shape=[n_out],stddev=initial_std_ndist), name='b_out')
        tf.summary.histogram('bout',bout)
        z_out = tf.matmul(layer_3,wout)+ bout
        tf.summary.histogram('z_out',z_out)
        layer_out = tf.reduce_sum(z_out,1,keepdims=True)
        tf.summary.tensor_summary('layer_out',layer_out)
        print(layer_out)

    with tf.variable_scope('loss'):
        cost = tf.reduce_mean(tf.square(layer_out-tf_y),name='cost')
        # loss = tf.losses.mean_squared_error(tf_y,z_out)

        tf.summary.scalar('cost',cost)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    train_opt = optimizer.minimize(cost)

    init = tf.global_variables_initializer()

    merged = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(logdir='./log/2feat_8_layer3_batch_500', graph=g)

with tf.Session(graph=g) as sess:
    sess.run(init)
    mini_batch_size = 100
    n_batch = X_train.shape[0] // mini_batch_size + (X_train.shape[0] % mini_batch_size != 0)

    fig, ax = plt.subplots()
    for epoch in range(80000):
        i_batch = (epoch % n_batch) * mini_batch_size
        batch = X_train[i_batch:i_batch + mini_batch_size], y_train[i_batch:i_batch + mini_batch_size]

        loss, summary, f_predict,_ = sess.run([cost, merged, layer_out, train_opt],feed_dict={tf_x:X_train,tf_y:y_train})
        # loss, summary, f_predict,_ = sess.run([cost, merged, layer_out, train_opt],feed_dict={tf_x:batch[0],tf_y:batch[1]})
        # loss = sess.run([loss],feed_dict={tf_x:X_train,tf_y:y_train})
        train_writer.add_summary(summary,epoch)
        print('epoch: {}, loss: {}'.format(epoch,loss))
        print(cost)
        if epoch % 100 ==99:
            # y_plot = y_train.reshape(1, -1)[0]
            pred_plot = f_predict.reshape(1, -1)[0]
            ax.clear()
            ax.plot(X_train, y_train[:])
            ax.plot(X_train, f_predict, 'g--')
            ax.set(xlabel='X Value', ylabel='Y / Predicted Value', title=[str(epoch)])
            plt.pause(0.001)

    plt.show()

