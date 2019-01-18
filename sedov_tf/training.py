import tensorflow as tf
import pandas as pd
from data_selection.data_processor import data_processor
import numpy as np

# path to the dataset
path = '../sedov_dataset/datasets/data_gamma_1.2_2.0_included'
# load the dataset into pandas dataframe
data_from_file = pd.read_csv(path)
# initialize data processor to work on dataframe
processor = data_processor()
# the values of gamma used in the dataset
gammas = processor.unique(data_from_file, 'gamma')
# values of energy used in dataset
energies = processor.unique(data_from_file, 'initial_energy')
data_from_file = processor.select(data_from_file,gamma=gammas[:-1])
# select the data for training
data = processor.select(data_from_file, gamma=gammas[::2])
# remove the colomn that contain the values for the density
data = processor.rmv_col(data, ['rho_max'])
# shuffle the dataset
data = data.reindex(np.random.permutation(data.index))
# select the features (X_train) and the lable (y_train)
X_train = data.loc[:,['arrival_distance','u_max','p_max','gamma']].values
y_train = data.loc[:,['initial_energy']].values

# standard deviation for normal distribution initialization of weights and biases
initial_std_ndist = 0.03

def build_graph(n_inputs,n_layer1,n_layer2, n_layer3,n_out,learning_rate, logdir):
    g = tf.Graph()

    with g.as_default():

        # create a place holder for the input value x
        tf_x = tf.placeholder(dtype=tf.float32, shape=[None,n_inputs], name='tf_x')
        print(tf_x)
        tf_y = tf.placeholder(dtype=tf.float32, shape=[None,n_out],name='tf_y')

        # building layer 1
        with tf.variable_scope(name_or_scope='layer1'):
            w1 = tf.Variable(tf.random_normal(shape=[n_inputs, n_layer1], stddev=initial_std_ndist), name='w_h1')
            tf.summary.histogram('w1',w1)
            b1 = tf.Variable(tf.random_normal(shape=[n_layer1],stddev=initial_std_ndist), name='b_h1')
            tf.summary.histogram('b1',b1)
            z_in = tf.matmul(tf_x,w1) + b1
            tf.summary.histogram('z_in',z_in)
            layer_1 = tf.nn.tanh(z_in)
            print(layer_1)
        # building layer 2
        with tf.variable_scope(name_or_scope='layer2'):
            w2 = tf.Variable(tf.random_normal(shape=[n_layer1, n_layer2], stddev=initial_std_ndist), name='w_h2')
            tf.summary.histogram('w2',w2)
            b2 = tf.Variable(tf.random_normal(shape=[n_layer2], stddev=initial_std_ndist), name='b_h2')
            tf.summary.histogram('b2',b2)
            z_h1 = tf.matmul(layer_1,w2) + b2
            tf.summary.histogram('z_h1',z_h1)
            layer_2 = tf.nn.tanh(z_h1)
            print(layer_2)
        # building layer 3
        with tf.variable_scope(name_or_scope='layer3'):
            w3 = tf.Variable(tf.random_normal(shape=[n_layer2, n_layer3], stddev=initial_std_ndist), name='w_h3')
            tf.summary.histogram('w3',w3)
            b3 = tf.Variable(tf.random_normal(shape=[n_layer3], stddev=initial_std_ndist), name='b_h3')
            tf.summary.histogram('b3',b3)
            z_h2 = tf.matmul(layer_2,w3) + b3
            tf.summary.histogram('z_h2',z_h2)
            layer_3 = tf.nn.tanh(z_h2)
            print(layer_3)
        # building output layer
        with tf.variable_scope(name_or_scope='layer_out'):
            wout = tf.Variable(tf.random_normal(shape=[n_layer3, n_out], stddev=initial_std_ndist), name='w_out')
            tf.summary.histogram('wout',wout)
            bout = tf.Variable(tf.random_normal(shape=[n_out],stddev=initial_std_ndist), name='b_out')
            tf.summary.histogram('bout',bout)
            z_out = tf.matmul(layer_3,wout)+ bout
            tf.summary.histogram('z_out',z_out)
            layer_out = tf.reduce_sum(z_out,1,keepdims=True, name='layer_out_op')
            tf.summary.tensor_summary('layer_out',layer_out)
            print(layer_out)
        # defining the cost function
        with tf.variable_scope('loss'):
            cost = tf.reduce_mean(tf.square(layer_out-tf_y),name='cost')
            # loss = tf.losses.mean_squared_error(tf_y,z_out)

            tf.summary.scalar('cost',cost)
        # initializing the  optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        # define the operation for optimizing
        train_opt = optimizer.minimize(cost)
        # define the global valrable initializer
        init = tf.global_variables_initializer()
        # define a merger to do the symmary
        merged = tf.summary.merge_all()
        # define a writer to wirte the summary into a file
        train_writer = tf.summary.FileWriter(logdir='./log/'+logdir, graph=g)
        # define a saver to save the model graph and paramters
        saver = tf.train.Saver()

    # start a session with graph g defined earlier
    with tf.Session(graph=g) as sess:
        sess.run(init) # initialize the variables
        mini_batch_size = 100   # select the size of the batch
        n_batch = X_train.shape[0] // mini_batch_size + (X_train.shape[0] % mini_batch_size != 0)

        # fig, ax = plt.subplots()
        # go over the data in the dataset for 100K time
        for epoch in range(100000):
            i_batch = (epoch % n_batch) * mini_batch_size
            batch = X_train[i_batch:i_batch + mini_batch_size], y_train[i_batch:i_batch + mini_batch_size]

            # loss, summary, f_predict,_ = sess.run([cost, merged, layer_out, train_opt],feed_dict={tf_x:X_train,tf_y:y_train})
            # run the session and collect the value of the loss, summary and the output
            loss, summary, f_predict,_ = sess.run([cost, merged, layer_out, train_opt],feed_dict={tf_x:batch[0],tf_y:batch[1]})
            # loss = sess.run([loss],feed_dict={tf_x:X_train,tf_y:y_train})
            if epoch % 1000 == 999:
                # print the cost and save the symmary into a log file every 1000 epoch
                train_writer.add_summary(summary,epoch)
                print('epoch: {}, loss: {}'.format(epoch,loss))
        # save the model parameters into model directory
        save_path = saver.save(sess, "./model/"+logdir+"/"+logdir+".ckpt")
        print("Model saved in path: %s" % save_path)



n1 = 10 # number of units in hidden layer 1
n2 = 8  # number of units in hidden layer 2
n3 = 4  # number of units in hidden layer 3
# name of the model
logdir = 'initial_energy_model_2_lr_1e-2_tanh_Adam'

#n_inputs is the number of features
# n_out is the number of outputs
build_graph(n_inputs= 4, n_layer1=n1, n_layer2=n2, n_layer3 = n3,n_out=1, learning_rate=0.01, logdir=logdir)
