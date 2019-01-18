import tensorflow as tf
import matplotlib.pyplot as plt

import pandas as pd
from data_selection.data_processor import data_processor
import numpy as np

path = '../sedov_dataset/datasets/data_gamma_1.2_2.0_included'
path_test = '../sedov_dataset/datasets/data_gamma_1.2_2.0_included'
test_data_from_file = pd.read_csv(path_test)
data_from_file = pd.read_csv(path)
processor = data_processor()
gammas = processor.unique(data_from_file, 'gamma')
energies = processor.unique(data_from_file, 'initial_energy')
data_from_file = processor.select(data_from_file,gamma=gammas[:-1])
data = processor.select(data_from_file, gamma=gammas[1::2])
data = processor.rmv_col(data, ['rho_max'])
data = data.reindex(np.random.permutation(data.index))
data_train = processor.select(data_from_file, gamma=gammas[::2])
data_train = processor.rmv_col(data_train, ['rho_max'])
data_train = data_train.reindex(np.random.permutation(data_train.index))

X_test = data.loc[:,['arrival_distance','u_max','p_max','gamma']].values
y_test = data.loc[:,['initial_energy']].values
X_train = data_train.loc[:,['arrival_distance','u_max','p_max','gamma']].values
y_train = data_train.loc[:,['initial_energy']].values

with tf.Session() as sess:

    saver = tf.train.import_meta_graph('./log_layers/model_relu/model_layers.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./log_layers/model_relu/'))

    g = tf.get_default_graph()

    tf_x = g.get_tensor_by_name('tf_x:0')
    predict = g.get_tensor_by_name('predict/Identity:0')

    # layer_out = g.get_tensor_by_name('predict/kernel:0')
    # print(layer_out)
    predicted_val = sess.run([predict], feed_dict={'tf_x:0':X_test})
    predicted_val_train = sess.run([predict], feed_dict={'tf_x:0':X_train})


plt.scatter(y_train[:],predicted_val_train[0][:], color = 'blue', edgecolors='w',label='training samples')
plt.scatter(y_test[:],predicted_val[0][:], color = 'green', marker='s', edgecolors='w', label='testing samples')
plt.legend()
plt.xlabel('exact')
plt.ylabel('predicted')
plt.savefig('./predict_vs_exact_tanh_layers.png')
plt.show()
