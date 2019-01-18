import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib
import pandas as pd
from data_selection.data_processor import data_processor
from plotters.random_forest_plotter import RF_plotter

# set the font size for the plots
matplotlib.rcParams.update({'font.size': 14})

# set the maximum rows to display for pandas dataframe
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.2f}'.format

# path for the dataset that you need to use
path = '../sedov_dataset/datasets/data_gamma_1.2_2.0_included'
# load the data into a pandas dataframe
data_from_file = pd.read_csv(path)

# initialize a processor to work with data
processor = data_processor()

# shuffle the dataset
data_from_file = data_from_file.reindex(np.random.permutation(data_from_file.index))

# the values of gamma used in this dataset
gammas = processor.unique(data_from_file, 'gamma')
print(gammas)

# collect the data for the features and target from the dataset
y, X = processor.feature_matrix_target(data_from_file,['gamma','arrival_distance','p_max','u_max'],'initial_energy')
print(X.shape)
print(X)
print(y.shape)
print(y)

# split the data for training and testing with testing size of 30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 123)
print(X_train)
print(y_train)

#n_jobs = -1 means all the CPUs
random_forest = RandomForestRegressor(random_state=123, max_features=int(4/3), n_jobs=-1, n_estimators=100)
print(random_forest)

# fitting the training data
random_forest.fit(X_train,y_train)

# initialize a plotter to build the component for the plots
plotter = RF_plotter(X_train,y_train, model=random_forest)

output_plot_path = './'
fig = plotter.predict_vs_exact(X_train, X_test, y_train, y_test, 'all', 'initial_energy')
plt.tight_layout()
plt.show()
print("done!!")
