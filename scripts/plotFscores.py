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

if len(sys.argv) <= 2:
    print "\n\tUsage: python plot.py $directory with csv files from analysis$ $id of the database$\n"
    sys.exit()

directory = sys.argv[1]
name = sys.argv[2]

algorithms = ["original", "unbalanced", "artificial", "smote"]
labels = ["Original", "Desbalanced", "SMOTE"]
color = ["b", "k", "r", "g"]
descriptors = ["BIC", "GCH", "CCV", "Haralick6", "ACC", "LBP", "HOG", "Contour", "Fisher"]
quantizations = ["Intensity", "Gleam", "Luminance", "MSB", "MSBModified", "BGR", "HSV"]
operations = ["Replicação", "Todos", "Borramento", "Mistura", "UnsharpMasking", "Composição de 16 quadrantes", "Combinação de Thresholds", "Saliência", "Smote Visual", "Ruído", "Composição de 4 quadrantes"]
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

        for generationType in range(1,11):
            fileName = directory + "/experiment-" + str(experiment) + "/analysis/" + descriptor + "_" + quantization + "_artificial_" + str(generationType) + "_FScore.csv"

            if os.path.exists(fileName):
                csvData = np.loadtxt(open(fileName, "rb"), delimiter=",", skiprows=0)
                generated_data[generationType].append(csvData)

    data = pd.DataFrame({
        'Unbalanced' : pd.Series(unbalanced_data, dtype='float'),
        'Smote' : pd.Series(smote_data, dtype='float'),
        # 'Artificial - 0: '+operations[0] : pd.Series(generated_data[0], dtype='float'),
        'Artificial - 1: '+operations[1] : pd.Series(generated_data[1], dtype='float'),
        'Artificial - 2: '+operations[2] : pd.Series(generated_data[2], dtype='float'),
        'Artificial - 3: '+operations[3] : pd.Series(generated_data[3], dtype='float'),
        'Artificial - 4: '+operations[4] : pd.Series(generated_data[4], dtype='float'),
        'Artificial - 5: '+operations[5] : pd.Series(generated_data[5], dtype='float'),
        'Artificial - 6: '+operations[6] : pd.Series(generated_data[6], dtype='float'),
        'Artificial - 7: '+operations[7] : pd.Series(generated_data[7], dtype='float'),
        'Artificial - 8: '+operations[8] : pd.Series(generated_data[8], dtype='float'),
        'Artificial - 9: '+operations[9] : pd.Series(generated_data[9], dtype='float'),
        'Artificial -10: '+operations[10] : pd.Series(generated_data[10], dtype='float'),
    })
    print data
    # print data.describe().transpose()[['mean', 'std', 'max', 'min']]
    # print data.describe().transpose()[['mean', 'std']]
    p = data.plot(kind='box', vert=False)
    fig = p.get_figure()
    fig.savefig(directory+descriptor+'_'+quantization+'_'+name+'.png', bbox_inches='tight')
