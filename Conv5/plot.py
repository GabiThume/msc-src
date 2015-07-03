    # coding: utf-8
#
# Visualização dos resultados gerados
#
# Para executar: python plot.py
#

from matplotlib import pyplot as plt
import numpy as np
import sys
import os

directory = "Analysis/"
algorithms = ["original","cnn", "smote"]
descriptors = ["BIC", "GCH", "CCV", "Haralick6", "ACC", "LBP", "HOG", "Contour"]
# descriptors = {"HOG"}
methods = ["Intensity", "Gleam", "Luminance", "MSB"]
operations = ["Replication", "ALL", "Blur", "Noise", "Blending", "UnsharpMasking", "Composition", "ThresholdCombination", "Saliency"]

which = 0

# for which in range(0,2):
for desc in descriptors:
    for met in methods:
        plt.clf()
        plt.title(desc+"_"+met)
        color = ["r", "g", "b"]
        labels = ["Original", 'Imagens Artificiais', 'SMOTE']
        plt.xlabel(u'Número de exemplos de treinamento da classe minoritária')

        if which == 0:
            plt.ylabel("% FScore")
        else:
            plt.ylabel("% Balanced Accuracy")

        xlim, ylim, xlimMin, ylimMin = 0, 0, 999, 999

        allAccuracy = []
        for i in range(0, 3):
            alg = directory+algorithms[i]+"_"
            if which == 0:
                fileName = alg+desc+"_"+met+"_FScore.csv"
            else:
                fileName = alg+desc+"_"+met+"_BalancedAccuracy.csv"
            print fileName
            if os.path.exists(fileName):
                csvData = np.loadtxt(open(fileName, "rb"), delimiter=",", skiprows=0)
                csvData = [[int(x), np.nanmean([b[1] for b in csvData if b[0] == x])] for x in np.unique(csvData[:, :1])]
                csvData = np.array(csvData)
                samples = csvData[:, 0]
                accuracy = csvData[:, 1]
                # accuracy = [(accuracy[len(accuracy)-1]-item) for item in accuracy]
                samples = [12, 25, 50]
                # allAccuracy.append(100)
                for item in np.array(accuracy):
                    allAccuracy.append(item)
                allAccuracy.append(0)
                line1, = plt.plot(samples, accuracy, color[i]+'o', label=labels[i], linewidth=3)
                plt.plot(samples, accuracy, color[i]+'-', linewidth=3)
                if max(samples) > xlim:
                    xlim = np.ceil(max(samples))
                if max(accuracy) > ylim:
                    ylim = np.ceil(max(accuracy))
                if min(samples) < xlimMin:
                    xlimMin = np.round(min(samples))
                if min(accuracy) < ylimMin:
                    ylimMin = np.round(min(accuracy))

        if os.path.exists(fileName):
            plt.legend(loc=3)

            plt.xticks(samples)
            plt.yticks(np.ceil(allAccuracy), rotation=0)
            plt.grid()
            ax = plt.axes()

            ax.tick_params(axis='y', labelsize=8)
            plt.gca().invert_xaxis()
            plt.gca().set_aspect('equal', adjustable='box')

            if which == 0:
                plt.savefig(directory+"graficos/"+desc+"_"+met+"_FScore.png")
            else:
                plt.savefig(directory+"graficos/"+desc+"_"+met+"_BalancedAccuracy.png")
