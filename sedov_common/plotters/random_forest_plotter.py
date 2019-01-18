import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


class RF_plotter(object):

    def __init__(self,features,target,model):
        '''

        :param features: the feature for the samples: numpy vector
        :param target: target values for the samples: numpy vector
        :param model: the Decision tree model
        '''
        self.features = features
        self.target = target
        self.model = model

    def scatter_2d(self):
        '''
        plots the scattered data with the fitting curve in 2-D target function of feature
        :return: matplotlib figure
        '''
        fig = plt.figure()
        indx_features = self.features.flatten().argsort()
        plt.scatter(self.features,self.target, c ='steelblue', edgecolors='white', s= 70)
        plt.plot(self.features[indx_features],self.model.predict(self.features[indx_features]), color='black', lw =2)
        r2 = r2_score(self.target, self.model.predict(self.features))
        return fig, r2

    def scatter_3d(self):
        '''
        plots the scattered data with the fitting surface in 3-D target function of two feature
        :return: matplotlib figure, ax ,and R2 value
        '''
        xx, yy = np.meshgrid(np.arange(self.features[:, 0].min(), self.features[:, 0].max(), 0.01),
                             np.arange(self.features[:, 1].min(), self.features[:, 1].max(), 0.01))

        X_model = np.array([xx.ravel().tolist(), yy.ravel().tolist()]).T

        zz = self.model.predict(X_model).reshape(xx.shape)

        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(self.features[:, 0], self.features[:, 1], self.target, c='steelblue', marker='o',
                   label='Dataset')
        r2 = r2_score(self.target,self.model.predict(self.features))
        ax.plot_surface(xx, yy, zz, color='green', alpha=0.5)

        return fig, ax, r2

    def residual(self,X_train,X_test,y_train,y_test,feature, target):
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

        # plt.scatter(y_train_pred, ((y_train_pred - y_train) / y_train) * 100, c='steelblue', marker='o',
        #             edgecolors='white', label=
        #             'training data')
        plt.scatter(y_train_pred, (y_train_pred - y_train), c='steelblue', marker='o',
                    edgecolors='white', label=
                    'training data')

        # plt.scatter(y_test_pred, ((y_test_pred - y_test) / y_test) * 100, c='limegreen', marker='s', edgecolors='white',
        #             label='test data')
        plt.scatter(y_test_pred,(y_test_pred - y_test), c='limegreen', marker='s', edgecolors='white',
                    label='test data')

        MSE_train = mean_squared_error(y_train,y_train_pred)

        MSE_test = mean_squared_error(y_test,y_test_pred)

        plt.xlabel('y_predict')

        plt.ylabel('Residual')

        plt.title('feature(' + feature + ') target(' + target + ')\n' + 'MSE_train: {:1.3E}'.format(MSE_train)
                  + '      MSE_test: {:1.3E}'.format(MSE_test))

        plt.axhline(y=0.0, color='black', linestyle='-')


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

    def feature_importance(self,feat_labels):
        # importance of features

        importances = self.model.feature_importances_

        indices = np.argsort(importances)[::-1]

        sorted_feature_labels = [feat_labels[i] for i in indices]

        feat_imp = {}
        for f in range(len(feat_labels)):
            string = "%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]])
            print(string)
            feat_imp[feat_labels[indices[f]]]= importances[indices[f]]

        fig, ax = plt.subplots()

        bars = ax.bar(range(len(feat_labels)),importances[indices]*100,align = 'center')

        plt.xticks(range(len(feat_labels)),sorted_feature_labels,rotation = 90)

        plt.xlim([-1,len(feat_labels)])
        plt.ylim([0, 110])

        def autolabel(rects):
            """
            Attach a text label above each bar displaying its height
            """
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                        '{:.1f}'.format(height),
                        ha='center', va='bottom')

        autolabel(bars)

        return fig, ax, feat_imp

    def MSE (self,X_train,X_test,y_train,y_test):

        y_train_pred = self.model.predict(X_train)

        y_test_pred = self.model.predict(X_test)

        r2 = r2_score(y_test, y_test_pred)

        MSE_train = mean_squared_error(y_train, y_train_pred)

        MSE_test = mean_squared_error(y_test, y_test_pred)

        return MSE_train, MSE_test, r2