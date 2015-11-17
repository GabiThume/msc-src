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
from bokeh.palettes import brewer
from scipy.interpolate import interp1d

features_file = sys.argv[1]
images_dir = sys.argv[2]

output_server("animated")
# output_file("vis.html");
image_size = 300
square_size = 10
alpha = 0.9

df = pd.io.parsers.read_csv(filepath_or_buffer = features_file, header=None, sep = ' ')
df.dropna(how="all", inplace=True, axis=1)
df.columns = ['Image label'] + ['Class label'] + ['Train or test'] + [l for l in range(3, df.shape[1])]
all_samples = df[range(3, df.shape[1])].values
classes = df['Class label'].values

sklearn_pca = sklearnPCA(n_components=2)
X_pca = sklearn_pca.fit_transform(all_samples)
p = figure(title = "artificial generation", x_range=[min(X_pca[:,0]), max(X_pca[:,0])], y_range=[min(X_pca[:,1]), max(X_pca[:,1])])

num_classes = len(np.unique(classes))
label_dict = {0: 'Elefante', 1: 'Cavalo'}

trainOrTest = df['Train or test'].values
treino = trainOrTest == 1
teste = trainOrTest == 2

generated = (classes==1)*treino
generated[100] = False
original = all_samples

classes_samples = []
for i in range(num_classes):
    classes_samples.append({'treino': (classes==i)*treino*~generated, 'teste': (classes==i)*teste*~generated})

if num_classes < 3:
    colors = brewer["Spectral"][3]
else:
    colors = brewer["Spectral"][num_classes]

samples = []
for label in range(num_classes):
    for status in ('treino', 'teste'):
        samples.append([(classes_samples[label][status])[~generated],
                        colors[label],
                        label_dict[label]+" - "+status,
                        status+str(label)])

# def animated_generation(features):

sklearn_pca = sklearnPCA(n_components=2)
X_pca = sklearn_pca.fit_transform(original[~generated])
x_plot = X_pca[:,0]
y_plot = X_pca[:,1]

samples.append([[], colors[num_classes], "Artificiais", "Generated"])

for index, color, label, name in samples:
    p.square(x = list(x_plot[index]), y = list(y_plot[index]), color = color, legend = label, size = square_size, line_dash = [6, 3], name = name)

show(p)

# images = [images_dir+str(label)+"/treino/"+str(i)+".png" for i in range(0,6)]
# p.image_url(x = list(x_plot[treino[from_class]]-image_size/2), y = list(y_plot[treino[from_class]]+image_size/2), url=images, global_alpha = alpha, h = image_size, w = image_size)


def animate (x0, y0, x1, y1, step):
    x = x0*(x0<x1) + x1*(x1<x0) + abs(x1-x0)/step
    y = y0 + (y1-y0)*((x-x0)/(x1-x0))
    return x, y

samples[-1][0] = len(X_pca)
original = original[~generated]
for image in generated.nonzero()[0]:
    sklearn_pca = sklearnPCA(n_components=2)
    original = np.append(original, [all_samples[image]], axis = 0)
    X_pca = sklearn_pca.fit_transform(original)

    for step in range (10, 2, -2):
        ds = []
        for index, color, label, name in samples:
            data = p.select(name = name)[0].data_source
            x0 = data.data['x']
            y0 = data.data['y']
            if name is "Generated":
                x1 = X_pca[:,0][index :]
                y1 = X_pca[:,1][index :]

                if len(x1)!=len(x0):
                    x1 = x1[:len(x1)-1]
                    y1 = y1[:len(y1)-1]

            else:
                x1 = X_pca[:,0][index]
                y1 = X_pca[:,1][index]

            x, y = x0, y0
            x, y = animate (x, y, x1, y1, step)
            data.data['x'] = list(x)
            data.data['y'] = list(y)
            ds.append(data)

            if name is "Generated":
                data.data['x'] = list(X_pca[:,0][index :])
                data.data['y'] = list(X_pca[:,1][index :])
            else:
                data.data['x'] = list(x1)
                data.data['y'] = list(y1)

            ds.append(data)
        cursession().store_objects(*ds)

    # img_file = [images_dir+str(label)+"/treino/"+str(image)+".png" ]
    # print img_file
    # p.image_url(x = X_pca[:,0][-1]-image_size/2, y = X_pca[:,1][-1]+image_size/2, url=img_file, global_alpha = alpha, h = image_size, w = image_size)
    #
    # time.sleep(.10)

# sklearn_lda = LDA(n_components=2)
# X_lda = sklearn_lda.fit_transform(X,y)
#
# plot_scikit(X_lda, 'LDA')
# show(p)

# animated_generation(X)
