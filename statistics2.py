from matplotlib import pyplot as plt
import numpy as np
import sys
import os

directory = "Desbalanced/analysis/"
algorithms = ["original", "smote", "artificial"]
descriptors = {"BIC", "GCH", "CCV", "Haralick6", "ACC"}
methods = {"Intensity", "Gleam", "Luminance", "MSB"}

for descriptor in descriptors:
    for operation in range(0,5):
        output = open(directory+"statistics_"+str(operation)+"_"+descriptor+".csv", "w")
        output.write("Data-set,Original,Smote,Artificial\n")
        for method in methods:
            row = []
            treina = [[25], [12], [6]]
            for tech in range(0, 3):
                line = []
                fileName = directory+str(operation)+"-"+algorithms[tech]+"_"+descriptor+"_"+method+"_FScore.csv"
                if os.path.exists(fileName):
                    csvData = np.loadtxt(open(fileName, "rb"), delimiter=",", skiprows=0)
                    csvData = [[int(x), np.nanmean([b[1] for b in csvData if b[0] == x])] for x in np.unique(csvData[:, :1])]
                    csvData = np.array(csvData)
                    line = csvData[:, 1][:-1]
                    row.append(line)
                    treina[0].append(line[2])
                    treina[1].append(line[1])
                    treina[2].append(line[0])
            if os.path.exists(fileName):
                for item in treina:
                    output.write(descriptor+method+str(int(item[0])))
                    for pos in range(1,len(item)):
                        output.write(","+str(item[pos]))
                    output.write("\n")
