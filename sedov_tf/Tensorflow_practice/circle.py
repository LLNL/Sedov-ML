import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import  shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
x1 = np.arange(0.1, 10, 0.01)
x2 = np.arange(0.1, 10, 0.01)

y_exact = np.sin(x1)

plt.plot(x1,y_exact)
# plt.show()
# y_exact = x1**2 + x2**2

dict = {'x1':x1,'x2':x2,'y':y_exact}

data = pd.DataFrame(dict)
data = shuffle(data)
X = data.loc[:,['x1']].values
# X = data.loc[:,['x1','x2']].values
y = data.loc[:,['y']].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=123)

print(X_train.shape,X_test.shape, y_train.shape,y_test.shape)



## defining some parameters
learning_rate = 0.0001
training_epochs = 1000
batch_size = 100

# network parameters
n_inputs = 1
n_hidden1 = 6
n_hidden2 = 3
n_outputs = 1

g = tf.Graph()

with g.as_default():

    tf_x = tf.placeholder(dtype=tf.float32, shape=[None,n_inputs], name='tf_x')
    print(tf_x)
    tf_y = tf.placeholder(dtype=tf.float32, shape=[None,n_outputs], name='tf_y')
    print(tf_y)

    # store weights and bias
    weights = {
        'h1':tf.Variable(tf.random_normal([n_inputs,n_hidden1],stddev=0.03)),
        'h2':tf.Variable(tf.random_normal([n_hidden1,n_hidden2],stddev=0.03)),
        'out':tf.Variable(tf.random_normal([n_hidden2,n_outputs],stddev=0.03))
    }
    biases ={
        'b1':tf.Variable(tf.random_normal([n_hidden1])),
        'b2':tf.Variable(tf.random_normal([n_hidden2])),
        'out':tf.Variable(tf.random_normal([n_outputs])),
    }
    print (weights)
    print(biases)

    layer_1 = tf.add(tf.matmul(tf_x,weights['h1']),biases['b1'])  # net input to the hidden layer 2
    activation_1 = tf.nn.tanh(layer_1)
    layer_2 = tf.add(tf.matmul(activation_1,weights['h2']),biases['b2'])  # net input to the hidden layer 2
    activation_2 = tf.nn.sigmoid(layer_2)
    layer_out = tf.add(tf.matmul(activation_2,weights['out']),biases['out'])

    print(activation_1)
    print(activation_2)
    print(layer_out)

    sqr_cost = tf.square(tf_y - layer_out,name='sqr_cost')

    mean_cost = tf.reduce_mean(sqr_cost, name='mean_cost')

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate,name='GradientDescent')

    print(optimizer)

    train_op = optimizer.minimize(sqr_cost)

    init = tf.global_variables_initializer()


with tf.Session(graph=g) as sess:
    sess.run(init)
    avg_cost = []

    # training cycle
    for epoch in range (training_epochs):

        _, c = sess.run([train_op,mean_cost],feed_dict={tf_x:X_train,
                                                   tf_y:y_train})
        avg_cost.append(c)
        print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(c))

    print('Optimization finished!!')

plt.plot([i for i in range (len(avg_cost))],avg_cost)
plt.show()

print(weights.values())