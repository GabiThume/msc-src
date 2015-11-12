from __future__ import division

import numpy as np
import pandas as pd
import csv
import math
from bokeh.plotting import output_file, figure, show
from sklearn.preprocessing import LabelEncoder
from sklearn.lda import LDA
from matplotlib import pyplot as plt
import sys

# H = 10
# W = 10
#
# p = figure(x_range=[0,10], y_range=[0,10])
#
# url = ["../test/Lenna.png"]
# # p.image_url(x=list(range(0,100,10)), y=list(range(0,100,10)), url=url, global_alpha=0.2, h = 10, w = 10)
# p.image_url(x=0, y=0, url=url, h=H, w=W)
#
# # Open in a browser
# show(p)

df = pd.io.parsers.read_csv(filepath_or_buffer = sys.argv[1], header=None, sep = ' ')
df.dropna(how="all", inplace=True, axis=1)
df.columns = ['Image label'] + ['Class label'] + ['Train or test'] + [l for l in range(3, df.shape[1])]
X = df[range(3, df.shape[1])].values
y = df['Class label'].values

num_classes = 2
label_dict = {0: 'Praia', 1: 'Montanha'}

trainOrTest = df['Train or test'].values

p = figure(x_range=[0,1000], y_range=[0,1000])

def plot_scikit(X, title):

    ax = plt.subplot(111)
    for label,marker,color in zip(range(2),('^', 'o'),('blue', 'red')):
        images = ["images/original/"+str(label)+"/"+str(i)+".png" for i in range(50)]
        from_class = y == label
        treino = trainOrTest == 1
        teste = trainOrTest == 2
        x_plot = X[:,0][from_class]
        y_plot = X[:,1][from_class]
        if label == 1:
            plt.scatter(x_plot[treino[from_class]], y_plot[treino[from_class]], marker=marker, color=color, alpha=0.5, label='treino')
            # print "treino", x_plot
            plt.scatter(x_plot[teste[from_class]], y_plot[teste[from_class]], marker=marker, color="yellow", alpha=0.5, label='teste')
            # print "teste", x_plot
            plt.scatter(x_plot[treino[from_class]][6:], y_plot[treino[from_class]][6:], marker=marker, color='green', alpha=0.5, label='generated')
        else:
            plt.scatter(x_plot, y_plot, marker=marker, color=color, alpha=0.5, label=label_dict[label])

        p.image_url(x = x_plot, y = y_plot, url=images, global_alpha=0.2, h = 10, w = 10)

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title(title)

    plt.grid()
    plt.tight_layout
    plt.show()

show(p)
# sklearn_lda = LDA(n_components=2)
# X_lda = sklearn_lda.fit_transform(X,y)
#
# plot_scikit(X_lda, 'LDA')

from sklearn.decomposition import PCA as sklearnPCA

sklearn_pca = sklearnPCA(n_components=2)
X_pca = sklearn_pca.fit_transform(X)

plot_scikit(X_pca, 'PCA')
