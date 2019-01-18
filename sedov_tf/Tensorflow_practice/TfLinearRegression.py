import tensorflow as tf
import numpy as np


class TfLinreg(object):
    def __init__(self,x_dim, learning_rate = 0.01, random_seed = None):

        self.x_dim = x_dim
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.g = tf.Graph()

        # build the model
        with self.g.as_default():

            ## set the graph-level random seed
            tf.set_random_seed = self.random_seed

            self.build()

            ## create initializer
            self.init_op = tf.global_variables_initializer()


    def build(self):

        ## define the place holders
        self.x = tf.placeholder(dtype=tf.float32, shape = (None, self.x_dim), name = 'x_input')
        self.y = tf.placeholder(dtype=tf.float32, shape = (None), name = 'y_input')

        print(self.x)
        print(self.y)

        ## define weight matrix and bias vector
        w = tf.Variable(tf.zeros(shape= (1)), name='weight')
        b = tf.Variable(tf.zeros(shape= (1)), name='bias')

        print(w)
        print(b)

        # calculate the net input
        self.z_net = tf.squeeze(w*self.x +b, name='z_net')
        print(self.z_net)

        sqr_error = tf.square(self.y - self.z_net, name='sqr_error')
        print(sqr_error)

        self.mean_cost = tf.reduce_mean(sqr_error, name='mean_cost')

        # define an optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate, name='GradientDescent')

        self.optimizer = optimizer.minimize(self.mean_cost)


def train_linreg(sess, model, X_train, y_train, num_epochs = 10):
    ## initialize all variables : w and b
    sess.run(model.init_op)

    training_cost = []

    for i in range(num_epochs):
        _, cost = sess.run([model.optimizer, model.mean_cost], feed_dict={model.x:X_train, model.y:y_train})
        training_cost.append(cost)

    return training_cost


# the features in the columns and the smaples in the rows. here we have 10 samples and one feature
x_train = np.arange(10).reshape((10,1))
y_train = np.array([1.0,1.3,3.1,2.0,5.0,6.3,6.6,7.4,8.0,9.0])

lrmodel = TfLinreg(x_dim=x_train.shape[1],learning_rate=0.01,random_seed=1)

sess = tf.Session(graph=lrmodel.g)

training_costs = train_linreg(sess,lrmodel, x_train, y_train, 10)

import matplotlib.pyplot as plt

plt.plot(range(1,len(training_costs)+1), training_costs)
plt.xlabel('Epochs')
plt.ylabel('training cost')
plt.tight_layout()
plt.show()