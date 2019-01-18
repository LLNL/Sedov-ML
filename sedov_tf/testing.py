import tensorflow as tf
import matplotlib.pyplot as plt

import pandas as pd
from data_selection.data_processor import data_processor
import numpy as np


path = '../sedov_dataset/datasets/data_gamma_1.2_2.0_included'
data_from_file = pd.read_csv(path)
processor = data_processor()
gammas = processor.unique(data_from_file, 'gamma')
energies = processor.unique(data_from_file, 'initial_energy')
data_from_file = processor.select(data_from_file,gamma=gammas[:-1])
# selecting the testing data and training data
data_test = processor.select(data_from_file, gamma=gammas[1::2])
data_train = processor.select(data_from_file, gamma=gammas[::2])
# remove the density
data_test = processor.rmv_col(data_test, ['rho_max'])
data_train = processor.rmv_col(data_train, ['rho_max'])
# shuffle the data
data_train = data_train.reindex(np.random.permutation(data_train.index))
data_test = data_test.reindex(np.random.permutation(data_test.index))

# select the features and the label
X_train = data_train.loc[:,['arrival_distance','u_max','p_max','gamma']].values
y_train = data_train.loc[:,['time']].values

X_test = data_test.loc[:,['arrival_distance','u_max','p_max','gamma']].values
y_test = data_test.loc[:,['time']].values

with tf.Session() as sess:
    # load the graph of the model we want
    # .meta is the file that represent the graph
    saver = tf.train.import_meta_graph('./model/initial_energy_model_2_lr_1e-2_tanh_Adam/initial_energy_model_2_lr_1e-2_tanh_Adam.ckpt.meta')
    # load the last checkpoint in the in the modle folder that contain the values of the weights and biases
    saver.restore(sess,tf.train.latest_checkpoint('./model/initial_energy_model_2_lr_1e-2_tanh_Adam/'))

    g = tf.get_default_graph()
    # get the tensor from the graph in the name of layer_out_op
    layer_out = g.get_tensor_by_name('layer_out/layer_out_op:0')
    print(layer_out)
    # predict the testing data
    # run the session to get the result of the layer_out by feeding it the the values of the samples features
    predicted_val = sess.run([layer_out], feed_dict={'tf_x:0':X_test})
    # predict the training data
    predicted_val_train = sess.run([layer_out], feed_dict={'tf_x:0':X_train})


print(predicted_val[0].shape)
# plot the predict vs exact
plt.scatter(y_train[::10],predicted_val_train[0][::10], color = 'blue', edgecolors='w',label='training samples')
plt.scatter(y_test[::10],predicted_val[0][::10], color = 'green', marker='s', edgecolors='w',label='testing samples')
plt.legend()
plt.xlabel('exact')
plt.ylabel('predicted')
# save the current plot in the local directory.
plt.savefig('./predict_vs_exact_time.png')
plt.show()
