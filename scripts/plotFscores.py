# coding: utf-8
#
# Run: python plot.py
#

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import numpy as np
import sys
import os
import itertools

if len(sys.argv) <= 1:
    print "\n\tUsage: python plot.py $directory with csv files from analysis$\n"
    sys.exit()

directory = sys.argv[1]

algorithms = ["original", "unbalanced", "artificial", "smote"]
labels = ["Original", "Desbalanced", "SMOTE"]
color = ["b", "k", "r", "g"]
descriptors = ["BIC", "GCH", "CCV", "Haralick6", "ACC", "LBP", "HOG", "Contour", "Fisher"]
quantizations = ["Intensity", "Gleam", "Luminance", "MSB", "MSBModified", "BGR", "HSV"]
operations = ["Replication", "ALL", "Blur", "Blending", "UnsharpMasking", "Composition", "ThresholdCombination", "Saliency", "VisualSmote", "Noise", "VisualSmote", "Composition", "Composition", "Composition", "Composition", "Composition", "Noise"]
measurements = ["FScore", "BalancedAccuracy"]
marker = itertools.cycle(('+', 'o', '^', 's'))
plt.figure(figsize=(10,10))

descriptors = ["BIC"]
quantizations = ["Intensity"]
# For each descriptor
for descriptor in descriptors:
  # For each method for reduce image colors
  for quantization in quantizations:
    fscores = []
    unbalanced_data = []
    smote_data = []
    generated_data = defaultdict(list)
    for experiment in range(0,40):

        fileName = directory + "/experiment-" + str(experiment) + "/analysis/" + descriptor + "_" + quantization + "_unbalanced_FScore.csv"

        if os.path.exists(fileName):
            csvData = np.loadtxt(open(fileName, "rb"), delimiter=",", skiprows=0)
            unbalanced_data.append(csvData)

        fileName = directory + "/experiment-" + str(experiment) + "/analysis/" + descriptor + "_" + quantization + "_smote_FScore.csv"

        if os.path.exists(fileName):
            csvData = np.loadtxt(open(fileName, "rb"), delimiter=",", skiprows=0)
            smote_data.append(csvData)

        for generationType in range(0,10):
            fileName = directory + "/experiment-" + str(experiment) + "/analysis/" + descriptor + "_" + quantization + "_artificial_" + str(generationType) + "_FScore.csv"

            if os.path.exists(fileName):
                csvData = np.loadtxt(open(fileName, "rb"), delimiter=",", skiprows=0)
                generated_data[generationType].append(csvData)

    data = pd.DataFrame({ 'unbalanced' : pd.Series(unbalanced_data, dtype='float'),
                          'smote' : pd.Series(smote_data, dtype='float'),
                          'artificial - 0' : pd.Series(generated_data[0], dtype='float'),
                          'artificial - 1' : pd.Series(generated_data[1], dtype='float'),
                          'artificial - 2' : pd.Series(generated_data[2], dtype='float'),
                          'artificial - 3' : pd.Series(generated_data[3], dtype='float'),
                          'artificial - 4' : pd.Series(generated_data[4], dtype='float'),
                          'artificial - 5' : pd.Series(generated_data[5], dtype='float'),
                          'artificial - 6' : pd.Series(generated_data[6], dtype='float'),
                          'artificial - 7' : pd.Series(generated_data[7], dtype='float'),
                          'artificial - 8' : pd.Series(generated_data[8], dtype='float'),
                          'artificial - 9' : pd.Series(generated_data[9], dtype='float'),
                        #   'artificial -10' : pd.Series(generated_data[10], dtype='int32'),
                        })
    print data.mean()
