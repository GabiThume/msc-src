from __future__ import division

import numpy as np
import pandas as pd
import csv
import math
import time
import sys

from bokeh.plotting import output_file, figure, show, output_server, cursession
from bokeh.palettes import brewer

from sklearn.preprocessing import LabelEncoder
from sklearn.lda import LDA
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.neighbors import KNeighborsClassifier

from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

def pca(samples):
    sklearn_pca = sklearnPCA(n_components=2)
    X_pca = sklearn_pca.fit_transform(samples)

    return (X_pca[:,0], X_pca[:,1])

def decision_region((x, y), classes, plot_size):
    # Plotting the decision region
    h = abs(x.max() - x.min())/100
    classifier = KNeighborsClassifier(n_neighbors = 1)
    classifier.fit( np.concatenate(([x], [y]), axis = 0).T, classes)
    if len(plot_size) != 4:
        minx = x.min()-1
        maxx = x.max()+1
        miny = y.min()-1
        maxy = y.max()+1
    else:
        minx, maxx, miny, maxy = plot_size
    xx, yy = np.meshgrid(np.arange(minx, maxx, h), np.arange(miny, maxy, h))
    region = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    return region.reshape(xx.shape)

def between_point(x0, y0, x1, y1, step):
    x = x0*(x0<x1) + x1*(x1<x0) + abs(x1-x0)/step
    y = y0 + (y1-y0)*((x-x0)/(x1-x0))
    return x, y


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
all_classes = df['Class label'].values
num_classes = len(np.unique(all_classes))
label_dict = {0: 'Elefante', 1: 'Cavalo'}

trainOrTest = df['Train or test'].values
treino = trainOrTest == 1
teste = trainOrTest == 2

generated = (all_classes==1)*treino # todo generated set
generated[100] = False
# Take only the original training samples
original = all_samples[~generated]# * treino]

if num_classes < 3:
    colors = brewer["Spectral"][3]
else:
    colors = brewer["Spectral"][num_classes]

x, y = pca(all_samples)
p = figure(title = "Geracao Artificial de Imagens", x_range=[min(x), max(x)], y_range=[min(y), max(y)])

region = decision_region((x, y), all_classes, ())
p.image(image=[region], x=[min(x)], y=[min(y)], dw=[abs(max(x)-min(x))], dh=[abs(max(y)-min(y))], palette="Greys3", alpha = 0.1, name = "region")
plot_size = region.shape

classes_samples = []
for i in range(num_classes):
    classes_samples.append({'treino': (all_classes==i)*treino,
                            'teste': (all_classes==i)*teste})

samples = []
for label in range(num_classes):
    for status in ('treino', 'teste'):
        samples.append([(classes_samples[label][status])[~generated],
                        colors[label],
                        label_dict[label]+" - "+status,
                        status+str(label)])

samples.append([[], colors[num_classes], "Artificiais", "Generated"])
for index, color, label, name in samples:
    p.square(x = list(x[index]), y = list(y[index]), color = color,
            legend = label, size = square_size, line_dash = [6, 3], name = name)

show(p)

# images = [images_dir+str(label)+"/treino/"+str(i)+".png" for i in range(0,6)]
# p.image_url(x = list(x_plot[treino[from_class]]-image_size/2), y = list(y_plot[treino[from_class]]+image_size/2), url=images, global_alpha = alpha, h = image_size, w = image_size)

samples[-1][0] = len(original)
classes = all_classes[~generated]
for image in generated.nonzero()[0]:
    original = np.append(original, [all_samples[image]], axis = 0)
    classes = np.append(classes, [all_classes[image]], axis = 0)
    x_plot, y_plot = pca(original)
    data = p.select(name = "region")[0].data_source
    region = decision_region((x_plot, y_plot), classes, plot_size)
    print region.shape
    # data.data['image'] =
    # cursession().store_objects(data)

    ds = []
    for index, color, label, name in samples:
        data = p.select(name = name)[0].data_source
        x0 = data.data['x']
        y0 = data.data['y']
        if name is "Generated":
            x1 = x_plot[index :]
            y1 = y_plot[index :]

            if len(x1) != len(x0):
                x1 = x1[:len(x1)-1]
                y1 = y1[:len(y1)-1]

        else:
            x1 = x_plot[index]
            y1 = y_plot[index]

        x, y = x0, y0
        x, y = between_point (x, y, x1, y1, 2)
        data.data['x'] = list(x)
        data.data['y'] = list(y)
        ds.append(data)

        if name is "Generated":
            data.data['x'] = list(x_plot[index :])
            data.data['y'] = list(y_plot[index :])
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
