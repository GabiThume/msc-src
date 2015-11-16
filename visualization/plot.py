from __future__ import division

import numpy as np
import pandas as pd
import csv
import math
import time
import sys

from bokeh.plotting import output_file, figure, show, output_server, cursession

from sklearn.preprocessing import LabelEncoder
from sklearn.lda import LDA

from matplotlib import pyplot as plt

from sklearn.decomposition import PCA as sklearnPCA

features_file = sys.argv[1]
images_dir = sys.argv[2]

output_server("animated")
# output_file("vis.html");
p = figure(plot_width=600, plot_height=600)

df = pd.io.parsers.read_csv(filepath_or_buffer = features_file, header=None, sep = ' ')
df.dropna(how="all", inplace=True, axis=1)
df.columns = ['Image label'] + ['Class label'] + ['Train or test'] + [l for l in range(3, df.shape[1])]
X = df[range(3, df.shape[1])].values
y = df['Class label'].values

num_classes = len(np.unique(y))
label_dict = {0: 'Elefante', 1: 'Cavalo'}

trainOrTest = df['Train or test'].values
treino = trainOrTest == 1
teste = trainOrTest == 2

generated = (y==1)*treino
generated[100] = False
original = ~generated

classes = []
for i in range(2):
    classes.append({'treino': (y==i)*treino*~generated, 'teste': (y==i)*teste*~generated})

image_size = 300
square_size = 10
alpha = 0.9

# def animated_generation(features):

sklearn_pca = sklearnPCA(n_components=2)
X = sklearn_pca.fit_transform(X)
x_plot = X[:,0]
y_plot = X[:,1]

for label,marker,color in zip(range(2),('^', 'o'),('blue', 'red')):
    if label == 1:
        # p.square(list(x_plot), list(y_plot), color = "Blue", legend= label_dict[label], size=square_size)
        images = [images_dir+str(label)+"/treino/"+str(i)+".png" for i in range(0,6)]
        p.square(x = list(x_plot[classes[label]['treino']]), y = list(y_plot[classes[label]['treino']]), color = "Red", legend= label_dict[label]+" - treino", size=square_size, line_dash = [6, 3], name = "treino"+str(label))
        # p.image_url(x = list(x_plot[treino[from_class]]-image_size/2), y = list(y_plot[treino[from_class]]+image_size/2), url=images, global_alpha = alpha, h = image_size, w = image_size)

        images = [images_dir+str(label)+"/teste/"+str(i)+".png" for i in range(50)]
        p.square(x = list(x_plot[classes[label]['teste']]), y = list(y_plot[classes[label]['teste']]), color = "Yellow", legend= label_dict[label]+" - teste", size=square_size, line_dash = [6, 3], name = "teste"+str(label))
        # p.image_url(x = list(x_plot[teste[from_class]]-image_size/2), y = list(y_plot[teste[from_class]]+image_size/2), url=images, global_alpha = alpha, h = image_size, w = image_size)
    else:
        images = [images_dir+str(label)+"/treino/"+str(i)+".png" for i in range(50)]
        p.square(x = list(x_plot[classes[label]['treino']]), y = list(y_plot[classes[label]['treino']]), color = "Purple", legend=label_dict[label], size=square_size, line_dash = [6, 3], name = "treino"+str(label))
        images = images + [images_dir+str(label)+"/teste/"+str(i)+".png" for i in range(50)]
        p.square(x = list(x_plot[classes[label]['teste']]), y = list(y_plot[classes[label]['teste']]), color = "Purple", legend=label_dict[label], size=square_size, line_dash = [6, 3], name = "teste"+str(label))
        # p.image_url(x = list(x_plot-image_size/2), y = list(y_plot+image_size/2), url=images, global_alpha = alpha, h = image_size, w = image_size)
show(p)


for image in range(len(generated)):
    sklearn_pca = sklearnPCA(n_components=2)
    original = np.append(X, [X[generated[image]]], axis = 0)
    X_pca = sklearn_pca.fit_transform(original)

    data = p.select(name = "treino1")[0].data_source
    data.data['x'] = list(X_pca[:,0][classes[0]['treino']])
    data.data['y'] = list(X_pca[:,1][classes[0]['treino']])
    cursession().store_objects(data)

    data = p.select(name = "teste1")[0].data_source
    data.data['x'] = list(X_pca[:,0][classes[1]['teste']])
    data.data['y'] = list(X_pca[:,1][classes[1]['teste']])
    cursession().store_objects(data)

    data = p.select(name = "treino0")[0].data_source
    data.data['x'] = list(X_pca[:,0][classes[0]['treino']])
    data.data['y'] = list(X_pca[:,1][classes[0]['treino']])
    cursession().store_objects(data)

    data = p.select(name = "teste0")[0].data_source
    data.data['x'] = list(X_pca[:,0][classes[0]['teste']])
    data.data['y'] = list(X_pca[:,1][classes[0]['teste']])
    cursession().store_objects(data)

    # img_file = [images_dir+str(label)+"/treino/"+str(image)+".png" ]
    # print img_file
    # p.square(X_pca[:,0][-1], X_pca[:,1][-1], color = "Black", legend= " - artificiais", size=square_size, line_dash = [6, 3], name = "generated"+str(image))
    # p.image_url(x = X_pca[:,0][-1]-image_size/2, y = X_pca[:,1][-1]+image_size/2, url=img_file, global_alpha = alpha, h = image_size, w = image_size)
    #
    # cursession().store_objects(p)
    # time.sleep(.10)

# sklearn_lda = LDA(n_components=2)
# X_lda = sklearn_lda.fit_transform(X,y)
#
# plot_scikit(X_lda, 'LDA')
# show(p)

# animated_generation(X)
