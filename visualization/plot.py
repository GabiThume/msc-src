from __future__ import division

import numpy as np
import pandas as pd
import csv
import math
import time
import sys

from bokeh.io import hplot, vform
from bokeh.plotting import figure, show, output_server, cursession, gridplot, output_file
from bokeh.palettes import brewer
from bokeh.models.widgets import Button, Tabs, Panel
from bokeh.properties import Instance

from sklearn.preprocessing import LabelEncoder
from sklearn.lda import LDA
from sklearn.decomposition import PCA as sklearnPCA, TruncatedSVD
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
        self.images = df['Image path'].values

        trainOrTest = df['Train or test'].values
        self.treino = trainOrTest == 1
        if self.treino.sum() == 0:
            self.treino = trainOrTest == 0
        self.teste = trainOrTest == 2

        self.generated = df['Generated'].values == True

        # Take only the original training samples
        self.original = ~self.generated * self.treino

        if self.num_classes < 3:
            self.colors = brewer["Spectral"][4]
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

        self.samples.append([
            [],
            self.colors[self.num_classes],
            "Artificiais",
            "Generated",
            "treino"
            ])

    def pca(self, samples):
        '''
        Apply pca from sklearn and return the two most important dimensions.
        '''
        # sklearn_pca = sklearnPCA(n_components=2)
        # # TruncatedSVD, whiten
        # X_pca = sklearn_pca.fit_transform(samples)
        #
        samples_mean = samples.mean(0)
        samples_center = samples - samples_mean
        eigenvectors, eigenvalues, vt = np.linalg.svd(samples_center)
        dataReduced = np.dot(np.transpose(eigenvectors), samples_center)
        return dataReduced

    def decision_region(self, features, classes):
        '''
        Return the decision region to plot on the figure. This region is decided
        using the 1-knn classifier.
        '''
        # Plotting the decision region
        # h = abs(x.max() - x.min())/300
        h = 100
        classifier = KNeighborsClassifier(n_neighbors = 1, weights = "distance")
        classifier.fit(features, classes)
        x, y = features[:,0], features[:,1]
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
        index_generation = len(current_samples)
        new_data = []
        for image in self.generated.nonzero()[0]:
            current_samples = np.append(current_samples, [self.all_samples[image]], axis = 0)
            curent_classes = np.append(curent_classes, [self.all_classes[image]], axis = 0)

            ds = []
            pca = self.pca(current_samples)
            x_plot, y_plot = (pca[:,0], pca[:,1])
            # data = p.select(name = "region")[0].data_source
            # region = self.decision_region(pca, curent_classes)
            # data.data['image'] = [region]
            # ds.append(data)

            for index, color, label, name, status in self.samples:
                if p.select(name = name) != []:
                    data = p.select(name = name)[0].data_source
                    x0 = data.data['x']
                    y0 = data.data['y']

                    if name is "Generated":
                        x1 = x_plot[index_generation :]
                        y1 = y_plot[index_generation :]

                        if len(x1) != len(x0):
                            x1 = x1[:len(x1)-1]
                            y1 = y1[:len(y1)-1]

                    else:
                        x1 = x_plot[index[generated_plot.original]]
                        y1 = y_plot[index[generated_plot.original]]

                    # x, y = x0, y0
                    # x, y = self.between_point ((x, y), (x1, y1), 2)
                    # data.data['x'] = list(x)
                    # data.data['y'] = list(y)
                    # ds.append(data)

                    if name is "Generated":
                        data.data['x'] = list(x_plot[index_generation :])
                        data.data['y'] = list(y_plot[index_generation :])
                    else:
                        data.data['x'] = list(x1)
                        data.data['y'] = list(y1)

                    ds.append(data)
            new_data.append(ds)
        return new_data
        # img_file = [images_dir+str(label)+"/treino/"+str(image)+".png" ]
        # print img_file
        # p.image_url(x = X_pca[:,0][-1]-image_size/2, y = X_pca[:,1][-1]+image_size/2, url=img_file, global_alpha = alpha, h = image_size, w = image_size)
        #


def visualization_original(generated_plot, title):
    # Initial setting using all samples less the generated ones
    # pca = generated_plot.pca(generated_plot.all_samples[~generated_plot.generated])
    # x, y = (pca[:,0], pca[:,1])

    # Use only original samples - not testing neither generated samples
    pca = generated_plot.pca(generated_plot.all_samples[generated_plot.original])
    x, y = (pca[:,0], pca[:,1])

    # TOOLS = 'box_zoom,box_select,crosshair,resize,reset'
    p = figure(title = title, x_range=[min(x), max(x)], y_range=[min(y), max(y)])
    # p.border_fill = "whitesmoke"
    p.legend.orientation = "bottom_right"
    # p.toolbar_location = "below"
    # region = generated_plot.decision_region(pca, generated_plot.all_classes[generated_plot.original])
    # p.image(image=[region], x=[min(x)], y=[min(y)], dw=[abs(max(x)-min(x))], dh=[abs(max(y)-min(y))], palette="Greys3", alpha = 0.1, name = "region")
    # plot_size = region.shape


    for index, color, label, name, status in generated_plot.samples:
        if status is "treino":
            if name is "Generated":
                x_plot = []
                y_plot = []

            else:
                print
                print len([index[generated_plot.original]])
                x_plot = x[index[generated_plot.original]]
                y_plot = y[index[generated_plot.original]]
            p.scatter(x = x_plot, y = y_plot , marker = "circle", color = color,
                    legend = label, size = square_size, line_dash = [6, 3], name = name)
            # p.image_url(x = list(x[index]-image_size/2), y = list(y[index]+image_size/2), url= ['../'+df['Image path'].values[~generated][index]], global_alpha = alpha, h = image_size, w = image_size)

    return p


def visualization_all(generated_plot, title):
    # Initial setting using all samples less the generated ones
    # pca = generated_plot.pca(generated_plot.all_samples[~generated_plot.generated])
    # x, y = (pca[:,0], pca[:,1])

    # Use only original samples - not testing neither generated samples
    pca = generated_plot.pca(generated_plot.all_samples[generated_plot.treino])
    x, y = (pca[:,0], pca[:,1])

    # TOOLS = 'box_zoom,box_select,crosshair,resize,reset'
    p = figure(title = title, x_range=[min(x), max(x)], y_range=[min(y), max(y)])
    # p.border_fill = "whitesmoke"
    p.legend.orientation = "bottom_right"
    # p.toolbar_location = "below"
    # region = generated_plot.decision_region(pca, generated_plot.all_classes[generated_plot.treino])
    # p.image(image=[region], x=[min(x)], y=[min(y)], dw=[abs(max(x)-min(x))], dh=[abs(max(y)-min(y))], palette="Greys4", alpha = 0.1, name = "region")
    # plot_size = region.shape


    for index, color, label, name, status in generated_plot.samples:
        print len(index), len(x), label, name
        if status is "treino":
            if name is "Generated":
                x_plot = x[generated_plot.generated[generated_plot.treino]]
                y_plot = y[generated_plot.generated[generated_plot.treino]]
                images = generated_plot.images[generated_plot.generated]

            else:
                x_plot = x[index[generated_plot.original]]
                y_plot = y[index[generated_plot.original]]
                images = generated_plot.images[index]
            p.scatter(x = x_plot, y = y_plot , marker = "circle", color = color,
                    legend = label, size = square_size, line_dash = [6, 3], name = name)
            # p.image_url(x = list(x_plot-image_size/2), y = list(y_plot+image_size/2), url= "0.png", global_alpha = 1, h = 1000, w = 1000)

    return p

output_server("animated")
# output_file("animated.html")
image_size = 300
square_size = 10
alpha = 0.9

# features_file = sys.argv[1]
# Plot original balanced class
features_file = "../data/elefante-cavalo/features/BIC_Intensity_64c_100r_200i_original.csv"
original_plot = plot(features_file, {0: 'Elefante', 1: 'Cavalo'})
original = visualization_original(original_plot, "Original")

# Plot desbalanced class
features_file = "../data/elefante-cavalo/Artificial/3-Rebalanced0/features/BIC_Intensity_64c_100r_200i_artificial.csv"
unbalanced_plot = plot(features_file, {0: 'Elefante', 1: 'Cavalo'})
unbalanced = visualization_original(unbalanced_plot, "Desbalanceado")

# Plot after images generation
features_file = "../data/elefante-cavalo/Artificial/3-Rebalanced0/features/BIC_Intensity_64c_100r_200i_artificial.csv"
generated_plot = plot(features_file, {0: 'Elefante', 1: 'Cavalo'})
generated = visualization_all(generated_plot, "Geracao Artificial de Imagens")

# Plot after smote
features_file = "../data/elefante-cavalo/features/BIC_Intensity_200i_smote.csv"
smote_plot = plot(features_file, {0: 'Elefante', 1: 'Cavalo'})
smote = visualization_all(smote_plot, "SMOTE")

# Plot images generation testing images
features_file = "../data/elefante-cavalo/Artificial/3-Rebalanced0/features/BIC_Intensity_64c_100r_200i_artificial.csv"
generated_plot = plot(features_file, {0: 'Elefante', 1: 'Cavalo'})
generated_testing = visualization_all(generated_plot, "Geracao Artificial de Imagens")

# Plot smote testing images
features_file = "../data/elefante-cavalo/features/BIC_Intensity_200i_smote.csv"
smote_plot = plot(features_file, {0: 'Elefante', 1: 'Cavalo'})
smote_testing = visualization_all(smote_plot, "SMOTE")

plot = gridplot([[original, unbalanced], [generated, smote], [generated_testing, smote_testing]], toolbar_location="above")

show(plot)

# tab1 = Panel(child=p, title="Scatter")
# tab2 = Panel(child=p, title="Images")
# tabs1 = Tabs(tabs = [tab1, tab2])
# Animate artificial and smote generation
# generated_data = generated_plot.animate(generated)
# smote_data = smote_plot.animate(smote)
# for item in range(len(generated_data)):
#     cursession().store_objects(*generated_data[item])
#     cursession().store_objects(*smote_data[item])
#     time.sleep(.10)
