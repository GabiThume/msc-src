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

from sklearn.preprocessing import LabelEncoder, normalize
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
                "treino",
                "circle"
                ])
            self.samples.append([
                (self.all_classes == label) * self.teste,
                self.colors[label],
                self.label_dict[label]+" - teste",
                "teste"+str(label),
                "teste",
                "asterisk"
                ])

        self.samples.append([
            self.generated,
            self.colors[self.num_classes],
            "Artificiais",
            "Generated",
            "treino",
            "circle"
            ])

    def pca(self, samples):
        '''
        Apply pca from sklearn.
        '''
        sklearn_pca = sklearnPCA(n_components=2)
        # Fit the model with samples
        fit = sklearn_pca.fit(samples)
        # Apply the dimensionality reduction on samples
        pca = fit.transform(samples)
        return pca

    def decision_region(self, features, classes):
        '''
        Return the decision region to plot on the figure. This region is decided
        using the 1-knn classifier.
        '''
        # Plotting the decision region
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
        h = abs(x.max() - x.min())/100
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

            for index, color, label, name, status, marker in self.samples:
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

def visualization(samples, index, title, which, plot_region, plot_image):
    '''
        If which == 1: only trained and not generated samples
        If which == 2: only trained, testing and not generated samples
        If which == 3: only trained and generated samples
        If which == 4: only testing samples
        If which == 5: all samples
    '''

    sklearn_pca = sklearnPCA(n_components=2,  whiten=False)
    # Data normalization
    data = samples.all_samples
    # data = normalize(samples.all_samples.astype(np.float64))
    # Fit the model with samples
    fit = sklearn_pca.fit(data[index])
    # Apply the dimensionality reduction on samples
    pca = fit.transform(data)
    x, y = (pca[:,0], pca[:,1])

    # TOOLS = 'box_zoom,box_select,crosshair,resize,reset'
    p = figure(title = title, x_range=[min(x), max(x)], y_range=[min(y), max(y)])
    # p.border_fill = "whitesmoke"
    p.legend.orientation = "bottom_right"

    if plot_region:
        region = samples.decision_region(pca[index], samples.all_classes[index])
        p.image(image=[region], x=[min(x)], y=[min(y)], dw=[abs(max(x)-min(x))], dh=[abs(max(y)-min(y))], palette="Greys4", alpha = 0.1, name = "region")
        # plot_size = region.shape

    for index, color, label, name, status, marker in samples.samples:
        if (which == 1 or which == 3) and status is "teste":
            continue
        if which == 4 and status is "treino":
            continue
        if which == 1 or which == 2 or which == 4:
            index = [] if name is "Generated" else index
        if plot_image:
            images = samples.images[index]
            if images != []:
                if images[0] != 'smote':
                    p.scatter(x = x[index], y = y[index] , marker = "square", color = color,
                            legend = label, size = square_size, name = name)

                    p.image_url(x = list(x[index]-wimage/2), y = list(y[index]+himage/2), url= "../"+images, global_alpha = 0.9, h = himage, w = wimage)
        else:
            p.scatter(x = x[index], y = y[index] , marker = marker, color = color,
                    legend = label, size = square_size,name = name)


    return p

output_server("animated")
square_size = 20
alpha = 0.9

wimage = 384*4
himage = 256*4

# features_file = sys.argv[1]
original_file = "../data/elefante-cavalo/features/BIC_Intensity_64c_100r_200i_original.csv"
artificial_file = "../data/elefante-cavalo/Artificial/3-Rebalanced0/features/BIC_Intensity_64c_100r_200i_artificial.csv"
smote_file = "../data/elefante-cavalo/features/BIC_Intensity_200i_smote.csv"
original_plot = plot(original_file, {0: 'Elefante', 1: 'Cavalo'})
generated_plot = plot(artificial_file, {0: 'Elefante', 1: 'Cavalo'})
smote_plot = plot(smote_file, {0: 'Elefante', 1: 'Cavalo'})

# Plot original balanced class
original = visualization(original_plot, original_plot.original, "Original", 1, False, False)

# Plot desbalanced class
unbalanced = visualization(generated_plot, generated_plot.original, "Desbalanceado", 2, False, False)
unbalanced_training = visualization(generated_plot, generated_plot.original, "Desbalanceado - Treino", 1, False, False)
unbalanced_testing = visualization(generated_plot, generated_plot.original, "Desbalanceado - Teste", 4, False, False)

# Plot after images generation
generated_training = visualization(generated_plot, generated_plot.treino, "Geracao Artificial de Imagens - Treino", 3, False, False)
# Plot images generation testing images
generated_testing = visualization(generated_plot, generated_plot.treino, "Geracao Artificial de Imagens - Teste", 4, False, False)
# Plot images generation images
generated = visualization(generated_plot, generated_plot.treino, "Geracao Artificial de Imagens", 5, False, False)

# Plot after smote
smote_training = visualization(smote_plot, smote_plot.treino, "SMOTE - Treino", 3, False, False)
# Plot smote testing images
smote_testing = visualization(smote_plot, smote_plot.treino, "SMOTE - Teste", 4, False, False)
# Plot smote images
smote = visualization(smote_plot, smote_plot.treino, "SMOTE", 5, False, False)

scatter = gridplot([
    # [original, generated, smote],
    [unbalanced, unbalanced_training, unbalanced_testing],
    [generated, generated_training, generated_testing],
    [smote, smote_training, smote_testing]],
    toolbar_location="above")

# Plot desbalanced class
unbalanced = visualization(generated_plot, generated_plot.original, "Desbalanceado", 2, True, False)
unbalanced_training = visualization(generated_plot, generated_plot.original, "Desbalanceado - Treino", 1, True, False)
unbalanced_testing = visualization(generated_plot, generated_plot.original, "Desbalanceado - Teste", 4, True, False)

# Plot after images generation
generated_training = visualization(generated_plot, generated_plot.treino, "Geracao Artificial de Imagens - Treino", 3, True, False)
# Plot images generation testing images
generated_testing = visualization(generated_plot, generated_plot.treino, "Geracao Artificial de Imagens - Teste", 4, True, False)
# Plot images generation images
generated = visualization(generated_plot, generated_plot.treino, "Geracao Artificial de Imagens", 5, True, False)

# Plot after smote
smote_training = visualization(smote_plot, smote_plot.treino, "SMOTE - Treino", 3, True, False)
# Plot smote testing images
smote_testing = visualization(smote_plot, smote_plot.treino, "SMOTE - Teste", 4, True, False)
# Plot smote images
smote = visualization(smote_plot, smote_plot.treino, "SMOTE", 5, True, False)

region = gridplot([
    # [original, generated, smote],
    [unbalanced, unbalanced_training, unbalanced_testing],
    [generated, generated_training, generated_testing],
    [smote, smote_training, smote_testing]],
    toolbar_location="above")

# Plot desbalanced class
unbalanced = visualization(generated_plot, generated_plot.original, "Desbalanceado", 2, False, True)
unbalanced_training = visualization(generated_plot, generated_plot.original, "Desbalanceado - Treino", 1, False, True)
unbalanced_testing = visualization(generated_plot, generated_plot.original, "Desbalanceado - Teste", 4, False, True)

# Plot after images generation
generated_training = visualization(generated_plot, generated_plot.treino, "Geracao Artificial de Imagens - Treino", 3, False, True)
# Plot images generation testing images
generated_testing = visualization(generated_plot, generated_plot.treino, "Geracao Artificial de Imagens - Teste", 4, False, True)
# Plot images generation images
generated = visualization(generated_plot, generated_plot.treino, "Geracao Artificial de Imagens", 5, False, True)

# Plot after smote
smote_training = visualization(smote_plot, smote_plot.treino, "SMOTE - Treino", 3, False, True)
# Plot smote testing images
smote_testing = visualization(smote_plot, smote_plot.treino, "SMOTE - Teste", 4, False, True)
# Plot smote images
smote = visualization(smote_plot, smote_plot.treino, "SMOTE", 5, False, True)

plot_images = gridplot([
    # [original, generated, smote],
    [unbalanced, unbalanced_training, unbalanced_testing],
    [generated, generated_training, generated_testing],
    [smote, smote_training, smote_testing]],
    toolbar_location="above")

tab1 = Panel(child=original, title="Original")
tab2 = Panel(child=scatter, title="Scatter")
tab3 = Panel(child=region, title="Decision region")
tab4 = Panel(child=plot_images, title="Images")
tabs = Tabs(tabs = [tab1, tab2, tab3, tab4])
show(tabs)



# Animate artificial and smote generation
generated_animation = visualization(generated_plot, generated_plot.treino, "Geracao Artificial de Imagens - Treino", 1, False, False)
smote_animation = visualization(smote_plot, smote_plot.treino, "SMOTE - Treino", 1, False, False)
generated_data = generated_plot.animate(generated_animation)
smote_data = smote_plot.animate(smote_animation)

original = hplot(original, generated_animation, smote_animation)

for item in range(len(generated_data)):
    cursession().store_objects(*generated_data[item])
    cursession().store_objects(*smote_data[item])
    time.sleep(.10)
