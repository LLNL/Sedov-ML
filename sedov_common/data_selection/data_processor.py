import pandas as pd
import numpy as np


class data_processor (object):
    '''
    this class is used to select data from the dataset ( panda DataFrame ) based on the values of gamma, energy and time
    '''

    def __init__(self):
        self.tolerance = 1e-10

    def select(self,data,gamma=None,time=None,energy= None):
        '''
        this function return a subset of the data based on the trim condition specified
        :param gamma: list of the values of gamma
        :param time: list of time values
        :param energy: list of energy values
        :return: a panda data frame of the subset you select
        '''
        if time == None:
            time = self.unique(data,'time')
        if energy == None:
            energy = self.unique(data,'initial_energy')
        if gamma == None:
            gamma = self.unique(data,'gamma')

        selection = pd.DataFrame(columns=self.columns_titles(data))
        selection = selection.append( pd.DataFrame(data.loc[data['gamma'].isin(gamma) &
                                data['time'].isin(time) &
                                data['initial_energy'].isin(energy)], columns=self.columns_titles(data)))

        return selection

    def unique(self,data,column):
        """
        find the unique values in the column
        :param column: key of the column: string
        :return: a list of unique values in the column
        """
        return (np.unique(data[column].values)).tolist()

    def columns_titles(self,data):
        '''
        fin the columns title of a data frame
        :param data: pandas DataFrame
        :return: list of columns titles
        '''

        return data.columns.values.tolist()
    def rmv_col(self,data,cols):

        for col in cols:
            del data[col]

        return data

    def feature_matrix_target(self,data_frame,target,features):

        X = data_frame.loc[:, features].values

        y = data_frame.loc[:, target].values

        return X, y

    def nearest_val_indx(self,data_frame,feature,value):
        unique_values = self.unique(data_frame,feature)

        for num in unique_values:
            diff = abs(num-value)
            if diff < self.tolerance:
                return unique_values.index(num)

        return None






