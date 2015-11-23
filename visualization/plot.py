from __future__ import division

import numpy as np
import pandas as pd
import csv
import math
import time
import sys

from bokeh.io import hplot, vform
from bokeh.plotting import figure, show, output_server, cursession
from bokeh.palettes import brewer
from bokeh.models.widgets import Button
from bokeh.properties import Instance

from sklearn.preprocessing import LabelEncoder
from sklearn.lda import LDA
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.neighbors import KNeighborsClassifier

from matplotlib import pyplot as plt
from scipy.interpolate import interp1d


class plot(object):

    def __init__(self, csvName, label_dict):
        self.csvName = csvName
        self.plot_size = []
        self.label_dict = label_dict

        df = pd.io.parsers.read_csv(filepath_or_buffer = self.csvName, header=None, sep=',', skiprows=1)
        df.dropna(how="all", inplace=True, axis=1)
        df.columns = ['Image path'] + ['Class label'] + ['Train or test'] + ['Generated'] + [l for l in range(4, df.shape[1])]
        self.all_samples = df[range(4, df.shape[1])].values
        self.all_classes = df['Class label'].values
        self.num_classes = len(np.unique(self.all_classes))

        trainOrTest = df['Train or test'].values
        self.treino = trainOrTest == 1
        self.teste = trainOrTest == 2

        self.generated = df['Generated'].values == True

        # Take only the original training samples
        self.original = ~self.generated * self.treino

        if self.num_classes < 3:
            self.colors = brewer["Spectral"][3]
        else:
            self.colors = brewer["Spectral"][self.num_classes]

        self.samples = []
        for label in range(self.num_classes):
            self.samples.append([
                (self.all_classes == label) * self.treino * ~self.generated,
                self.colors[label],
                self.label_dict[label]+" - treino",
                "treino"+str(label),
                "treino"
                ])
            self.samples.append([
                (self.all_classes == label) * self.teste * ~self.generated,
                self.colors[label],
                self.label_dict[label]+" - teste",
                "teste"+str(label),
                "teste"
                ])


    def pca(self, samples):
        '''
        Apply pca from sklearn and return the two most important dimensions.
        '''
        sklearn_pca = sklearnPCA(n_components=2)
        X_pca = sklearn_pca.fit_transform(samples)

        return (X_pca[:,0], X_pca[:,1])

    def decision_region(self, (x, y), classes):
        '''
        Return the decision region to plot on the figure. This region is decided
        using the 1-knn classifier.
        '''
        # Plotting the decision region
        h = abs(x.max() - x.min())/100
        classifier = KNeighborsClassifier(n_neighbors = 1)
        classifier.fit( np.concatenate(([x], [y]), axis = 0).T, classes)
        if len(self.plot_size) != 4:
            minx = x.min()-1
            maxx = x.max()+1
            miny = y.min()-1
            maxy = y.max()+1
        else:
            minx, maxx, miny, maxy = self.plot_size
        xx, yy = np.meshgrid(np.arange(minx, maxx, h), np.arange(miny, maxy, h))
        region = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
        return region.reshape(xx.shape)

    def between_point(self, (x0, y0), (x1, y1), step):
        '''
        Return the (x,y) coordinate between the inputs coordinates.
        '''
        x = x0*(x0<x1) + x1*(x1<x0) + abs(x1-x0)/step
        y = y0 + (y1-y0)*((x-x0)/(x1-x0))
        return x, y

    def animate(self, p):
        current_samples = self.all_samples[self.original]
        curent_classes = self.all_classes[self.original]
        self.samples.append([
            [len(current_samples)],
            self.colors[self.num_classes],
            "Artificiais",
            "Generated",
            "treino"
            ])
        for image in self.generated.nonzero()[0]:
            current_samples = np.append(current_samples, [self.all_samples[image]], axis = 0)
            curent_classes = np.append(curent_classes, [self.all_classes[image]], axis = 0)

            x_plot, y_plot = self.pca(current_samples)
            data = p.select(name = "region")[0].data_source
            region = self.decision_region((x_plot, y_plot), curent_classes)
            data.data['image'] = [region]
            cursession().store_objects(data)

            ds = []
            for index, color, label, name, status in self.samples:
                if p.select(name = name) != []:
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
                        x1 = x_plot[index[generated_plot.original]]
                        y1 = y_plot[index[generated_plot.original]]

                    x, y = x0, y0
                    x, y = self.between_point ((x, y), (x1, y1), 2)
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



def visualization(generated_plot, title):
    # Initial setting using all samples less the generated ones
    x, y = generated_plot.pca(generated_plot.all_samples[~generated_plot.generated])
    # TOOLS = 'box_zoom,box_select,crosshair,resize,reset'
    p = figure(title = title, x_range=[min(x), max(x)], y_range=[min(y), max(y)])
    # p.border_fill = "whitesmoke"
    p.legend.orientation = "bottom_right"
    # p.toolbar_location = "below"
    region = generated_plot.decision_region((x, y), generated_plot.all_classes[~generated_plot.generated])
    p.image(image=[region], x=[min(x)], y=[min(y)], dw=[abs(max(x)-min(x))], dh=[abs(max(y)-min(y))], palette="Greys3", alpha = 0.1, name = "region")
    plot_size = region.shape

    # Use only original samples - not testing neither generated samples
    x, y = generated_plot.pca(generated_plot.all_samples[generated_plot.original])

    for index, color, label, name, status in generated_plot.samples:
        print len(index), len(x), label, name
        if status is "treino":
            print len(x), index[generated_plot.original]
            p.square(x = list(x[index[generated_plot.original]]), y = list(y[index[generated_plot.original]]), color = color,
                    legend = label, size = square_size, line_dash = [6, 3], name = name)
            # p.image_url(x = list(x[index]-image_size/2), y = list(y[index]+image_size/2), url= ['../'+df['Image path'].values[~generated][index]], global_alpha = alpha, h = image_size, w = image_size)

    return p


output_server("animated")
image_size = 300
square_size = 10
alpha = 0.9

# features_file = sys.argv[1]
features_file = "../data/elefante-cavalo/Artificial/3-Rebalanced0/features/BIC_Intensity_64c_100r_200i_artificial.csv"
generated_plot = plot(features_file, {0: 'Elefante', 1: 'Cavalo'})
p1 = visualization(generated_plot, "Geracao Artificial de Imagens")

features_file = "../data/elefante-cavalo/features/BIC_Intensity_200i_smote.csv"
smote_plot = plot(features_file, {0: 'Elefante', 1: 'Cavalo'})
p2 = visualization(smote_plot, "SMOTE")

show(hplot(p1, p2))

# show(p1)
generated_plot.animate(p1)
smote_plot.animate(p2)


# button = Button(label="Foo", type="success")
# button.on_change('selected', generate)
# show(hplot(p1, p2))
