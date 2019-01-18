import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from plotters.random_forest_plotter import RF_plotter
from data_selection.data_processor import data_processor
from sklearn.model_selection import train_test_split

# import data and

path = '../../sedov_dataset/datasets/data_extrapolate_at_gamma_1.2_2_0.02_all_energy'

data_from_file = pd.read_csv(path)

# initialize  a data processor
processor = data_processor()
# print(data_from_file.describe())
# reading the values of unique values of gammas and energies
gammas = processor.unique(data_from_file, 'gamma')
energies = processor.unique(data_from_file, 'initial_energy')
times = processor.unique(data_from_file,'time')
# print (gammas)
# print(energies)
# pre-processing the data
train_gamma = gammas.copy()
# del train_gamma[processor.nearest_val_indx(data_from_file,'gamma',1.4)]
# del train_gamma[processor.nearest_val_indx(data_from_file,'gamma',1.58)]
# fit=1.4
# ext=1.5
data = processor.select(data_from_file, gamma=train_gamma)
# data = processor.select(data_from_file, gamma=[train_gamma[processor.nearest_val_indx(data_from_file,'gamma',fit)]])
# print(data.describe())
data = processor.rmv_col(data, ['rho_max'])
data = data.reindex(np.random.permutation(data.index))

X,y = processor.feature_matrix_target(data,'initial_energy',['p_max','u_max','arrival_distance','gamma'])
# print(X)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)
# print(X_train)

n_features = 4
random_seed = 123
np.random.seed(random_seed)

g = tf.Graph()

with g.as_default():
    tf.set_random_seed(random_seed)
    tf_x = tf.placeholder(dtype=tf.float32, shape=(None,n_features), name='tf_x')
    tf_y = tf.placeholder(dtype=tf.float32, shape=None, name='tf_y')

    h1 = tf.layers.dense(inputs=tf_x,units=3,activation=tf.nn.relu)

    # h2 = tf.layers.dense(inputs=h1, units=10, activation=tf.nn.relu)

    output = tf.layers.dense(inputs=h1, units=1, activation=tf.nn.relu)

    # define the cost function and optimizer

    cost = tf.reduce_mean(tf.square(output-tf_y), name='cost')

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

    train_op = optimizer.minimize(loss=cost)

    init_op = tf.global_variables_initializer()


def create_batch_generator(X, y, batch_size=128, shuffle=False):
    X_copy = np.array(X)
    y_copy = np.array(y)

    if shuffle:
        data = np.column_stack((X_copy,y_copy))
        np.random.shuffle(data)
        X_copy = data[:,:-1]
        y_copy = data[:,-1]

    for i in range(0,X.shape[0],batch_size):
        yield (X_copy[i:i+batch_size,:], y_copy[i:i+batch_size])


# create a session
sess =tf.Session(graph=g)
sess.run(init_op)

for epoch in range(10):
    training_costs =[]
    batch_generator = create_batch_generator(X_train, y_train, batch_size=64)

    for batch_X, batch_y in batch_generator:
        # print('batch_X')
        # print(batch_X)
        feed = {tf_x:batch_X,tf_y:batch_y}

        _, c = sess.run([train_op,cost],feed_dict=feed)

        training_costs.append(c)

        print('Epoch %4d: %.4f'%(epoch,c))



## do predictions

feed = {tf_x:X_test,tf_y:y_test}

y_pred, cost = sess.run([output,cost],feed)

print(cost)
