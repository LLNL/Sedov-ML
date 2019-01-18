import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve, GridSearchCV
from sklearn.pipeline import make_pipeline
import numpy as np

class sensitivity(object):

    def __init__(self,model,X_train,y_train,cross_validation_folds,number_of_cores,scaling_model=None):

        self.model = model
        self.scaling_model = scaling_model
        self.X_train = X_train
        self.y_train = y_train
        self.cross_validation_folds = cross_validation_folds
        self.number_of_cores = number_of_cores

    def pipe_line_ (self):
        if self.scaling_model == None:
            return make_pipeline(self.model)
        else:
            return make_pipeline(self.scaling_model)


    def size_training (self, number_of_interval, log =False ):

        train_sizes, train_scores, test_scores = learning_curve(estimator=self.pipe_line_(),
                                                                X = self.X_train,
                                                                y = self.y_train,
                                                                train_sizes = np.linspace(0.1,1.0,number_of_interval),
                                                                cv = self.cross_validation_folds,
                                                                n_jobs=self.number_of_cores)

        train_mean = np.mean(train_scores,axis=1)
        train_std = np.std(train_scores,axis=1)
        test_mean = np.mean(test_scores,axis=1)
        test_std = np.std(test_scores,axis=1)

        fig = plt.figure()
        plt.plot(train_sizes, train_mean, color = 'blue', marker = 'o', markersize = 5, label = 'training accuracy')

        plt.fill_between(train_sizes,train_mean+train_std, train_mean - train_std, alpha = 0.15, color = 'blue')

        plt.plot(train_sizes, test_mean, color = 'green', linestyle = '--', marker = 's', markersize = 5, label = 'testing accuracy')

        # plt.fill_between(train_sizes, test_mean+ test_std, test_mean - test_std, alpha= 0.15 , color = 'green')

        plt.grid()

        if log:
            plt.xscale('log')

        plt.xlabel('Number samples')

        plt.ylabel('Accuracy')

        plt.legend(loc = 'lower right')

        return fig

    def hyper_param_validation_curve(self, param_name, param_range, log = False):

        train_scores, test_scores = validation_curve(estimator=self.pipe_line_(),
                                                     X=self.X_train,
                                                     y=self.y_train,
                                                     param_name = param_name,
                                                     param_range = param_range,
                                                     cv = self.cross_validation_folds)

        train_mean = np.mean(train_scores,axis =1)
        train_std = np.std(train_scores,axis =1)
        test_mean = np.mean(test_scores,axis =1)
        test_std = np.std(test_scores,axis =1)

        fig = plt.figure()

        plt.plot(param_range, train_mean,color = 'blue', marker='o', markersize=5, label = 'training accuracy')

        plt.fill_between(param_range, train_mean+train_std, train_mean- train_std,color = 'blue', alpha = 0.15 )

        plt.plot(param_range, test_mean,color = 'green', marker='s', linestyle = '--', markersize=5, label = 'testing accuracy')

        plt.fill_between(param_range, test_mean+test_std, test_mean- test_std,color = 'green', alpha = 0.15 )

        plt.grid()

        if log:
            plt.xscale('log')

        plt.legend(loc='lower right')

        plt.xlabel(param_name)

        plt.ylabel('Accuracy')

        return fig


    def tuning_hyperparameter_grid_search(self, param_grid):
        '''

        :param param_grid: is a list of dictionaries each dictionary corresponds to an estimator. each estimator vary
                            certain number of hyper-parameters
        :return: best parameter, best score, best estimator, best index
        '''

        gs = GridSearchCV(estimator=self.pipe_line_(),
                          param_grid=param_grid,
                          # scoring='accuracy',
                          cv = self.cross_validation_folds,
                          n_jobs= self.number_of_cores)

        gs = gs.fit(self.X_train,self.y_train)

        return gs.best_params_, gs.best_score_



