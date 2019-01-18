from tensorflow import keras as kr
import tensorflow as tf
import pandas as pd
from data_selection.data_processor import data_processor
import numpy as np

# path to the dataset
path = '../sedov_dataset/datasets/data_gamma_1.2_2.0_included'
# load the dataset into a pandas dataframe
data_from_file = pd.read_csv(path)
# initialize a data processor to work on the data in the dataframe
processor = data_processor()
# collect the values of gammas used in the dataset
gammas = processor.unique(data_from_file, 'gamma')
# collect the values of energies used in the dataset
energies = processor.unique(data_from_file, 'initial_energy')

data_from_file = processor.select(data_from_file, gamma=gammas[:-1])
# select the samples of the dataset to train on
# here we select 1.2, 1.22, 1.24 ... and we want to test on 1.21, 1.23, ...
data = processor.select(data_from_file, gamma=gammas[::2])
# remove the density column form the dataset
data = processor.rmv_col(data, ['rho_max'])
# shuffle the data
data = data.reindex(np.random.permutation(data.index))

# select the training feature (X_train) and label (y_train)
X_train = data.loc[:,['arrival_distance','u_max','p_max','gamma']].values
y_train = data.loc[:,['initial_energy']].values

print(X_train,y_train)

n1 = 15 # number of units in the first hidden layer
n2 = 10 # number of units in the second hidden layer
n3 = 6  # number of units in the third hidden layer

model = kr.models.Sequential()

model.add(
    kr.layers.Dense(units=n1,
                    input_dim=4,    # number of feature input
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros',
                    activation='tanh')
)

model.add(
    kr.layers.Dense(units=n2,
                    input_dim=n1,
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros',
                    activation='tanh'
    )
)

model.add(
    kr.layers.Dense(units=n3,
                    input_dim=n2,
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros',
                    activation='tanh')
)

model.add(
    kr.layers.Dense(units=1,    # number of output features
                    input_dim=n3,
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros',
                    activation='linear'
                    )
)

sgd_optimizer = kr.optimizers.SGD(lr=0.01, decay=1e-7, momentum=0.9)

model.compile(optimizer=sgd_optimizer, loss='mean_squared_error')
model.fit(x=X_train, y=y_train, batch_size=1000, epochs=1000, verbose=1)
