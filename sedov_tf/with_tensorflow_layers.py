import tensorflow as tf
import pandas as pd
from data_selection.data_processor import data_processor
import numpy as np

path = '../sedov_dataset/datasets/data_gamma_1.2_2.0_included'
data_from_file = pd.read_csv(path)
processor = data_processor()
gammas = processor.unique(data_from_file, 'gamma')
energies = processor.unique(data_from_file, 'initial_energy')
data_from_file = processor.select(data_from_file,gamma=gammas[:-1])

data = processor.select(data_from_file, gamma=gammas[::2])
data = processor.rmv_col(data, ['rho_max'])
data = data.reindex(np.random.permutation(data.index))

X_train = data.loc[:,['arrival_distance','u_max','p_max','gamma']].values
y_train = data.loc[:,['initial_energy']].values

initial_std_ndist = 0.03

n_inputs = 4
n_out = 1
n_layer1 = 10
n_layer2 = 8
n_layer3 = 4

g = tf.Graph()

with g.as_default():

    tf_x = tf.placeholder(dtype=tf.float32, shape=[None, n_inputs], name='tf_x')
    print(tf_x)
    tf_y = tf.placeholder(dtype=tf.float32, shape=[None, n_out], name='tf_y')
    print(tf_y)

    layer_1 = tf.layers.dense(inputs=tf_x, units=n_layer1, activation=tf.nn.relu, name='layer_1')
    #tf.contrib.layers.summarize_activation(layer_1)
    print(layer_1)
    layer_2 = tf.layers.dense(inputs=layer_1, units=n_layer2, activation=tf.nn.relu, name='layer_2')
    #tf.contrib.layers.summarize_activation(layer_2)
    print(layer_2)
    layer_3 = tf.layers.dense(inputs=layer_2, units=n_layer3, activation=tf.nn.relu, name='layer_3')
    #tf.contrib.layers.summarize_activation(layer_3)
    print(layer_3)

    predict = tf.layers.dense(inputs=layer_3, units=n_out, activation=tf.identity, name='predict')
    #tf.contrib.layers.summarize_activation(predict)
    print(predict)

    loss = tf.reduce_mean(tf.square(tf_y - predict), name='loss')
    tf.summary.scalar('loss',loss)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.25)

    train_op = optimizer.minimize(loss)

    merged = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(logdir='./log_layers/' + 'TF_layers_relu', graph=g)

    saver = tf.train.Saver()

    init = tf.global_variables_initializer()

with tf.Session(graph=g) as sess:
    sess.run(init)
    mini_batch_size = 100
    n_batch = X_train.shape[0] // mini_batch_size + (X_train.shape[0] % mini_batch_size != 0)

    # fig, ax = plt.subplots()
    for epoch in range(100000):
        i_batch = (epoch % n_batch) * mini_batch_size
        batch = X_train[i_batch:i_batch + mini_batch_size], y_train[i_batch:i_batch + mini_batch_size]

        cost,summary,_ = sess.run([loss,merged,train_op],feed_dict={tf_x:batch[0],tf_y:batch[1]})

        if epoch % 1000 == 999:
            train_writer.add_summary(summary,global_step=epoch)
            print('epoch {} : cost {}'.format(epoch,cost))

    saver.save(sess,'./log_layers/model_relu/model_layers.ckpt')


