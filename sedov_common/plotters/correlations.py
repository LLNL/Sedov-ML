import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix

class correlations(object):
     def __init__(self, data):

         self.data = data


     def matrix(self):
         ## Producing the correlation matrix for this data set

         cols = [index for index in self.data.keys()]

         # correlation coefficients matrix
         cm = np.corrcoef(self.data[cols].values.T)

         fig, ax = plt.subplots()
         im = ax.imshow(cm, alpha=0.5)

         # We want to show all ticks...
         ax.set_xticks(np.arange(len(cols)))
         ax.set_yticks(np.arange(len(cols)))
         # ... and label them with the respective list entries
         ax.set_xticklabels(cols)
         ax.set_yticklabels(cols)

         # Rotate the tick labels and set their alignment.
         plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                  rotation_mode="anchor")

         # Loop over data dimensions and create text annotations.
         for i in range(len(cols)):
             for j in range(len(cols)):
                 text = ax.text(j, i, '%.2f' % cm[i, j],
                                ha="center", va="center", color="k")

         ax.set_title("Correlation Matrix")

         return fig , ax


     def scatter(self):

         scatterMatrix = scatter_matrix(self.data, figsize=(12, 12), diagonal='hist')
         for ax in scatterMatrix.ravel():
             ax.set_xlabel(ax.get_xlabel(), fontsize=16, rotation=0)
             ax.set_ylabel(ax.get_ylabel(), fontsize=16, rotation=90)
             ax.set_yticklabels(ax.get_yticks(), fontsize=12)
             ax.set_xticklabels(ax.get_xticks(), fontsize=12)
             ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
             ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


         return scatterMatrix
