# coding: utf-8
#
# Run: python plot.py
#

import os
import sys
import numpy as np
import itertools
from matplotlib import pyplot as plt
from tabulate import tabulate

if len(sys.argv) <= 2:
    print "\n\tUsage: python tables_comparison.py $directory with csv files from analysis$ $database name$\n"
    sys.exit()

directory = sys.argv[1]
base = sys.argv[2]

descriptors   = ["BIC", "GCH", "CCV", "Haralick6", "ACC", "LBP", "HOG", "Contour", "Fisher"]
quantizations = ["Intensity", "Gleam", "Luminance", "MSB", "MSBModified"]
algorithms    = ["original", "desbalanced", "smote", "artificial"]
labels        = ["Original", "Desbalanced", "SMOTE"]
operations    = ["Replication", "ALL", "Blur", "Blending", "UnsharpMasking", "Composition", "ThresholdCombination", "Saliency", "VisualSmote", "Noise"]
measurements  = ["BalancedAccuracy", "FScore"]

for descriptor in descriptors:
    for quantization in quantizations:
        headers = [descriptor+"_"+quantization+"_"+base, "Balanced accuracy", "std", "diff", "FScore", "std", "diff"]
        table = []
        original = 0
        originalFscore = 0.0
        originalAcc = 0.0
        for i in range(0, len(algorithms)):
            csvMeanAcc = []; csvStdAcc = []; csvMeanFScore = []; csvStdFScore = []
            for operation in range(0, len(operations)):
                if i == 3:
                    csvMeanAcc = []; csvStdAcc = []; csvMeanFScore = []; csvStdFScore = []
                for measure in range (0, len(measurements)):
                    alg = directory+str(operation)+"-"+algorithms[i]+"_"
                    fileName = alg+descriptor+"_"+quantization+"_"+measurements[measure]+".csv"
                    if os.path.exists(fileName):
                        # print fileName
                        x, csvData = np.loadtxt(open(fileName, "rb"), delimiter=",", unpack=True)
                        if not (i == 2 and operation == 0):
                            if measure == 0:
                                csvMeanAcc.append(np.nanmean(csvData))
                                csvStdAcc.append(np.nanstd(csvData))
                            else:
                                csvMeanFScore.append(np.nanmean(csvData))
                                csvStdFScore.append(np.nanstd(csvData))
                if csvMeanAcc != [] and i == 3:
                    meanAcc = float(csvMeanAcc[0])
                    stdAcc = float(csvStdAcc[0])
                    meanFScore = float(csvMeanFScore[0])
                    stdFScore = float(csvStdFScore[0])
                    table.append([operations[operation], meanAcc, stdAcc, meanAcc-originalAcc, meanFScore, stdFScore, meanFScore-originalFscore])
            if csvMeanAcc != [] and i < 3:
                meanAcc = np.nanmean(csvMeanAcc)
                stdAcc = np.nanstd(csvMeanAcc)
                meanFScore = np.nanmean(csvMeanFScore)
                stdFScore = np.nanstd(csvMeanFScore)
                if i == original:
                    table.append([labels[i], meanAcc, stdAcc, 0.0, meanFScore, stdFScore, 0.0])
                    originalAcc = table[0][1]
                    originalFscore = table[0][4]
                else:
                    table.append([labels[i], meanAcc, stdAcc, meanAcc-originalAcc, meanFScore, stdFScore, meanFScore-originalFscore])
        if table !=[]:
            file_name = directory+descriptor+"-"+quantization+".txt"
            print file_name
            output = open(file_name, 'w+')
            tab = tabulate(table, headers, floatfmt=".2f")
            print tabulate(table, headers, floatfmt=".2f")
            output.write(tab)
            output.close()
