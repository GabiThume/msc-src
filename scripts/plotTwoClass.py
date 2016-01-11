# coding: utf-8
#
# Run: python plot.py
#

from matplotlib import pyplot as plt
import numpy as np
import sys
import os
import itertools


if len(sys.argv) <= 1:
    print "\n\tUsage: python plot.py $directory with csv files from analysis$\n"
    sys.exit()

directory = sys.argv[1]
algorithms = ["original", "unbalanced", "artificial", "smote"]
labels = ["Original", "Desbalanced", "Imagens Artificiais", "SMOTE"]
color = ["b", "k", "r", "g"]
descriptors = ["BIC", "GCH", "CCV", "Haralick6", "ACC", "LBP", "HOG", "Contour", "Fisher"]
methods = ["Intensity", "Gleam", "Luminance", "MSB", "MSBModified", "BGR", "HSV"]
operations = ["Replication", "ALL", "Blur", "Blending", "UnsharpMasking", "Composition", "ThresholdCombination", "Saliency", "VisualSmote", "Noise", "VisualSmote", "Composition", "Composition", "Composition", "Composition", "Composition", "Noise"]
measurements = ["FScore", "BalancedAccuracy"]
marker = itertools.cycle(('+', 'o', '^', 's'))
# marker = itertools.cycle(('s'))
plt.figure(figsize=(10,10))

for measure in measurements:
    for desc in descriptors:
        for met in methods:
            plt.clf()
            plt.title(desc+"_"+met)
            plt.xlabel(u'Operação de Geração Artificial')
            plt.ylabel("% "+measure)
            xlim, ylim, xlimMin, ylimMin = 0, 0, 999, 999
            allAccuracy = []
            for i in range(0, len(algorithms)):
                for operation in range(0,10):
                    accuracy = []
                    samples = []
                    for experiment in range(0,40):
                        alg = directory+"/experiment-"+str(experiment)+"/analysis/";
                        fileName = alg+desc+"_"+met+"_"+algorithms[i]+"_"+measure+".csv"

                        if not os.path.exists(fileName):
                            fileName = alg+desc+"_"+met+"_"+algorithms[i]+"_"+str(operation)+"_"+measure+".csv"

                        if os.path.exists(fileName):
                            csvData = np.loadtxt(open(fileName, "rb"), delimiter=",", skiprows=0)
                            print fileName, csvData
                            # if len(csvData) > 2:
                                # csvData = [b[1] for b in csvData]
                            # csvData = np.nanmean(csvData)
                            # else:
                            #     csvData = csvData[0]

                            # csvData = [[int(x), np.nanmean([b[1] for b in csvData if b[0] == x])] for x in np.unique(csvData[:, :1])]
                            # csvData = np.array(csvData)
                            samples.append(experiment)
                            # if i < 2:
                            # if (accuracy == []):
                            #     accuracy.append(csvData[1])
                            # else:
                            #     accuracy.append(accuracy[0])
                            # # else:
                            accuracy.append(csvData)
                            allAccuracy.append(csvData)
                            # samples = csvData[:, 0]
                            # accuracy = csvData[:, 1]
                            # accuracy = [(accuracy[len(accuracy)-1]-item) for item in accuracy]
                            # samples = [12, 25, 50]

                            # for item in np.array(accuracy):
                            #     allAccuracy.append(item)

                    if accuracy != []:
                        m = marker.next()
                        if algorithms[i] == "artificial":
                            line1, = plt.plot(samples, accuracy, color[i]+'o', label=labels[i]+str(operation), marker='s', alpha=.8)
                            plt.plot(samples, accuracy, color[i]+'o', marker='s', alpha=.5)
                        # if max(samples) > xlim:
                        #     xlim = np.ceil(max(samples))
                        # if max(accuracy) > ylim:
                        #     ylim = np.ceil(max(accuracy))
                        # if min(samples) < xlimMin:
                        #     xlimMin = np.round(min(samples))
                        # if min(accuracy) < ylimMin:
                        #     ylimMin = np.round(min(accuracy))


                if accuracy != []:
                    if algorithms[i] != "artificial":
                        line1, = plt.plot(samples, accuracy, color[i]+'o', label=labels[i], marker='s', alpha=.5)
                        plt.plot(samples, accuracy, color[i]+'o', marker='s', alpha=.5)

            if accuracy != []:
                plt.legend(loc=1, prop={'size':8})
                # plt.xticks(range(0,9))
                allAccuracy.append(0);
                allAccuracy.append(100);
                plt.yticks(np.ceil(allAccuracy), rotation=0)
                # plt.yticks(range(0,100))
                plt.grid()
                ax = plt.axes()
                ax.tick_params(axis='y', labelsize=7)
                # plt.gca().set_aspect('equal', adjustable='box')

                plt.savefig(directory+"/graficos/"+measure+"/"+desc+"_"+met+"_"+measure+".png",  dpi=100)
                print "Saved in: "+directory+"/graficos/"+measure+"/"+desc+"_"+met+"_"+measure+".png"
