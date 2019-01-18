import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

class L_plotter(object):

    def __init__(self, model, features, target, degree, resolution = 0.01):
        '''

        :param model: the regression model
        :param features: the feature for the samples: numpy vector
        :param target: target values for the samples: numpy vector
        :param degree: degree of polynomial fit : integer
        :param resolution: resolution of the fit curve or plane : float number
        '''
        # polynomial feature
        self.quadratic = PolynomialFeatures(degree=degree)
        self.model = model
        self.features = features
        self.features_quad = self.quadratic.fit_transform(self.features)
        self.target = target
        self.degree = degree
        self.resolution = resolution

    def scatter_2d(self):
        '''
        scatter_2d: plots the scattered data with the fitting curve in 2-D target function of feature
        :return: matplotlib figure
        '''
        feature_min = self.features.min()
        feature_max = self.features.max()



        X = np.arange(feature_min,feature_max,self.resolution)

        X_quad = self.quadratic.fit_transform(X.reshape(-1,1))

        predicted_target = self.model.predict(X_quad)

        fig = plt.figure()

        plt.scatter(self.features, self.target, c='steelblue', marker='o', edgecolors='white', label='Dataset')

        plt.plot(X, predicted_target, color='black', linewidth=2,
                 label='poly deg:'+str(self.degree)+
                       ', $R^2$=' + '{:.3f}'.format(r2_score(self.target, self.model.predict(self.features_quad))))

        return fig


    def scatter_3d(self):
        '''
        scatter_3d: plots the scattered data with the fitting surface in 3-D target function of two feature
        :return: matplotlib figure, ax ,and R2 value
        '''
        xx, yy = np.meshgrid(np.arange(self.features[:, 0].min(), self.features[:, 0].max(), self.resolution),
                             np.arange(self.features[:, 1].min(), self.features[:, 1].max(), self.resolution))

        X = np.array([xx.ravel().tolist(), yy.ravel().tolist()]).T

        X_quad = self.quadratic.fit_transform(X)

        zz = self.model.predict(X_quad).reshape(xx.shape)

        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(self.features[:, 0], self.features[:, 1], self.target, c='steelblue', marker='o', label='Dataset')

        ax.plot_surface(xx, yy, zz, color='green', alpha=0.5)

        r2 = r2_score(self.target, self.model.predict(self.features_quad))
        return fig, ax, r2


    def residual(self,X_train,X_test,y_train,y_test,feature,target):
        '''
        plot the residual as relative error
        :param X_train: training set of features
        :param X_test: testing set of features
        :param y_train: train set of target
        :param y_test: test set of target
        :param feature: feature name : string
        :param target: target name : string
        :return: matplotlib figure
        '''
        y_train_pred = self.model.predict(X_train)

        y_test_pred = self.model.predict(X_test)

        # residual plot where we simply subtract the true target variables from the predicted responses

        fig = plt.figure()

        plt.scatter(y_train_pred, ((y_train_pred - y_train) / y_train) * 100, c='steelblue', marker='o',
                    edgecolors='white', label=
                    'training data')

        plt.scatter(y_test_pred, ((y_test_pred - y_test) / y_test) * 100, c='limegreen', marker='s', edgecolors='white',
                    label='test data')

        MSE_train = mean_squared_error(y_train_pred,y_train_pred)

        MSE_test = mean_squared_error(y_test,y_test_pred)

        plt.xlabel('y_predict')

        plt.ylabel('Relative error (%)')

        plt.legend(loc='upper left')

        plt.title('feature(' + feature + ') target(' + target + ')\n'+'MSE_train: {:1.3E}'.format(MSE_train)
                  +'      MSE_test: {:1.3E}'.format(MSE_test))

        plt.axhline(y=0.0, color='black', linestyle='-')

        plt.tight_layout()

        return fig

    def predict_vs_exact(self,X_train,X_test,y_train,y_test,feature, target):

        y_train_pred = self.model.predict(X_train)

        y_test_pred = self.model.predict(X_test)

        r2 = r2_score(y_test, y_test_pred)

        fig = plt.figure()

        plt.scatter(y_train,y_train_pred, c='steelblue', marker='o',
                    edgecolors='white', label=
                    'training data')

        plt.scatter(y_test, y_test_pred, c='lightgreen', marker='s',
                    edgecolors='white', label=
                    'testing data')

        plt.xlabel('exact')

        plt.ylabel('predicted')

        MSE_train = mean_squared_error(y_train, y_train_pred)

        MSE_test = mean_squared_error(y_test, y_test_pred)

        plt.title('feature(' + feature + ') target(' + target + ')\n' + 'MSE_train: {:1.3E}'.format(MSE_train)
                  + '      MSE_test: {:1.3E}'.format(MSE_test)+'\n dataset size: training: '+str(y_train.shape[0])
                  + ' testing: '+str(y_test.shape[0])+'\n'+'R2: {:.3f}'.format(r2))

        return fig