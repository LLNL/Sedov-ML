import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from data_selection.data_processor import data_processor
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from plotters.Linear_reg_plotter import L_plotter

matplotlib.rcParams.update({'font.size': 14})

pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.2f}'.format


# load the data from the file:
path = '../sedov_dataset/datasets/data_gamma_1.2_2.0_included'
data_from_file = pd.read_csv(path)
processor = data_processor()
gammas = processor.unique(data_from_file, 'gamma')
energies = processor.unique(data_from_file, 'initial_energy')
times = processor.unique(data_from_file, 'time')

#  selecting data for fixed gamma 1.4 and fixed time equal to 1
data = processor.select(data_from_file, gamma=[gammas[processor.nearest_val_indx(data_from_file,'gamma',1.4)]],
                        time = [times[processor.nearest_val_indx(data_from_file,'time',1.0)]])
data = processor.rmv_col(data, ['rho_max'])

# shuffle data
data = data.reindex(np.random.permutation(data.index))

# collect the data for the features and target from the dataset
y, X = processor.feature_matrix_target(data,['gamma','arrival_distance','p_max','u_max'],'initial_energy')

# polynomial degree
degree = 2

quadratic = PolynomialFeatures(degree=degree)
# generating training set and testing set using feature cross
X_quad = quadratic.fit_transform(X)

# splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_quad, y, test_size=0.3, random_state=1)

# initialize the Regression model
slr = LinearRegression()
# fit the data
slr.fit(X_train, y_train)

#initialize the linear plotter
plotter_all = L_plotter(slr, X, y, degree)

fig = plotter_all.predict_vs_exact(X_train,X_test,y_train,y_test,'all','initial_energy')
plt.tight_layout()
plt.show()