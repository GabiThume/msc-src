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

features_file = sys.argv[1]
images_dir = sys.argv[2]

output_file("vis.html");

df = pd.io.parsers.read_csv(filepath_or_buffer = features_file, header=None, sep = ' ')
df.dropna(how="all", inplace=True, axis=1)
df.columns = ['Image label'] + ['Class label'] + ['Train or test'] + [l for l in range(3, df.shape[1])]
X = df[range(3, df.shape[1])].values
y = df['Class label'].values

num_classes = 2
label_dict = {0: 'Elefante', 1: 'Cavalo'}

trainOrTest = df['Train or test'].values

p = figure(plot_width=800, plot_height=800)

image_size = 300
square_size = 10
alpha = 0.9

def plot_scikit(X, title):

    ax = plt.subplot(111)
    for label,marker,color in zip(range(2),('^', 'o'),('blue', 'red')):
        from_class = y == label
        treino = trainOrTest == 1
        teste = trainOrTest == 2
        x_plot = X[:,0][from_class]
        y_plot = X[:,1][from_class]
        if label == 1:
            # p.square(list(x_plot), list(y_plot), color = "Blue", legend= label_dict[label], size=square_size)
            images = [images_dir+str(label)+"/treino/"+str(i)+".png" for i in range(0,6)]
            p.square(list(x_plot[treino[from_class]]), list(y_plot[treino[from_class]]), color = "Red", legend= label_dict[label]+" - treino", size=square_size, line_dash = [6, 3])
            p.image_url(x = list(x_plot[treino[from_class]]-image_size/2), y = list(y_plot[treino[from_class]]+image_size/2), url=images, global_alpha = alpha, h = image_size, w = image_size)

            images = [images_dir+str(label)+"/teste/"+str(i)+".png" for i in range(50)]
            p.square(list(x_plot[teste[from_class]]), list(y_plot[teste[from_class]]), color = "Yellow", legend= label_dict[label]+" - teste", size=square_size, line_dash = [6, 3])
            p.image_url(x = list(x_plot[teste[from_class]]-image_size/2), y = list(y_plot[teste[from_class]]+image_size/2), url=images, global_alpha = alpha, h = image_size, w = image_size)

            images = [images_dir+str(label)+"/treino/"+str(i)+".png" for i in range(6,50)]
            p.square(list(x_plot[treino[from_class]][6:]), list(y_plot[treino[from_class]][6:]), color = "Black", legend= label_dict[label]+" - artificiais", size=square_size, line_dash = [6, 3])
            p.image_url(x = list(x_plot[treino[from_class]][6:]-image_size/2), y = list(y_plot[treino[from_class]][6:]+image_size/2), url=images, global_alpha = alpha, h = image_size, w = image_size)
        else:
            images = [images_dir+str(label)+"/treino/"+str(i)+".png" for i in range(50)]
            images = images + [images_dir+str(label)+"/teste/"+str(i)+".png" for i in range(50)]
            p.square(list(x_plot), list(y_plot), color = "Purple", legend=label_dict[label], size=square_size, line_dash = [6, 3])
            p.image_url(x = list(x_plot-image_size/2), y = list(y_plot+image_size/2), url=images, global_alpha = alpha, h = image_size, w = image_size)

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title(title)

    plt.grid()
    plt.tight_layout
    # plt.show()

# sklearn_lda = LDA(n_components=2)
# X_lda = sklearn_lda.fit_transform(X,y)
#
# plot_scikit(X_lda, 'LDA')
# show(p)

from sklearn.decomposition import PCA as sklearnPCA

sklearn_pca = sklearnPCA(n_components=2)
X_pca = sklearn_pca.fit_transform(X)

plot_scikit(X_pca, 'PCA')
show(p)
