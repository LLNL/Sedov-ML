import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

# Model input and output
x = tf.placeholder(tf.float32, [None, 1])

# training data
x_plot = np.arange(0, math.pi*2*2, 0.1)
x_train = x_plot.reshape(-1, 1)
y_train_tf = tf.sin(x)

# Model parameters
W1 = tf.Variable(tf.random_normal([1,10], stddev=0.03), dtype=tf.float32, name='W1')
b1 = tf.Variable(tf.random_normal([10], stddev=0.03), dtype=tf.float32, name='b1')
W2 = tf.Variable(tf.random_normal([10,3], stddev=0.03), dtype=tf.float32, name='W2')
b2 = tf.Variable(tf.random_normal([3], stddev=0.03), dtype=tf.float32, name='b2')
W3 = tf.Variable(tf.random_normal([3,1], stddev=0.03), dtype=tf.float32, name='W3')
b3 = tf.Variable(tf.random_normal([1], stddev=0.03), dtype=tf.float32, name='b3')

layer1 = tf.tanh(tf.multiply(x,W1) + b1)
layer2 = tf.tanh(tf.matmul(layer1, W2) + b2)
linear_model = tf.reduce_sum(tf.matmul(layer2, W3) + b3, 1, keep_dims=True)

# loss
#loss = tf.reduce_sum(tf.square(linear_model - y_train_tf)) # sum of the squares
loss = tf.losses.mean_squared_error(y_train_tf,linear_model)

tf.summary.scalar('loss', loss)
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
# Merge all the summaries
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('train_tensorboard',sess.graph)

sess.run(init) # reset values to wrong

fig, ax = plt.subplots()

for i in range(40000):
    summary, f_predict, _ = sess.run([merged, linear_model, train], feed_dict={x: x_train})
    y_train, curr_layer1, curr_layer2, curr_W1, curr_b1, curr_W2, curr_b2, curr_W3, curr_b3, curr_loss = sess.run([y_train_tf,layer1, layer2, W1, b1, W2, b2, W3, b3, loss],
                                                                               {x: x_train})
    train_writer.add_summary(summary, i)
    if i % 1000 == 999:
        print("step ", i)
        print("W1: %s b1: %s" % (curr_W1, curr_b1))
        print("W2: %s b2: %s" % (curr_W2, curr_b2))
        print("W3: %s b3: %s" % (curr_W3, curr_b3))
        print("layer1: %s layer2: %s" % (curr_layer1, curr_layer2))
        print("linear_model: %s loss: %s" % (f_predict, curr_loss))
        print(" ")
        y_plot = y_train.reshape(1, -1)[0]
        pred_plot = f_predict.reshape(1, -1)[0]
        plt.hold(False)
        ax.plot(x_plot, y_train[:])
        plt.hold(True)
        ax.plot(x_plot, f_predict, 'g--')
        ax.set(xlabel='X Value', ylabel='Y / Predicted Value', title=[str(i)," Loss: ", curr_loss])
        plt.pause(0.001)

fig.savefig("fig1.png")
plt.show()