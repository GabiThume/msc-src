from matplotlib import pyplot as plt
import numpy as np

which = 1

files = ["Desbalanced/analysis/original_accuracy_", "Desbalanced/analysis/smote_accuracy_", "Desbalanced/analysis/rebalance_accuracy_"]
# meanAccuracy = ["Desbalanced/analysis/original_accuracy.csv", "Desbalanced/analysis/smote_accuracy.csv", "Desbalanced/analysis/rebalance_accuracy.csv"]

descriptors = {"BIC", "GCH", "CCV", "Haralick6", "ACC"}
methods = {"Intensity", "Gleam", "Luminance", "MSB"}

# descriptors = {"Haralick6"}
# methods = {"MSB"}

for desc in descriptors:
    for met in methods:
        plt.clf()
        plt.title(desc+"_"+met)
        color = ["r", "g", "b"]
        labels = ["Original", "SMOTE", "Artificial Images"]
        plt.xlabel("Number of minority samples")

        if which == 0:
            plt.ylabel("% Minority Correct")
        else:
            plt.ylabel("% Balanced Accuracy")

        xlim, ylim, xlimMin, ylimMin = 0, 0, 999, 999

        for i in range(0, 3):
            if which == 0:
                fileName = files[i]+desc+"_"+met+"_TPR.csv"
            else:
                fileName = files[i]+desc+"_"+met+"_BalancedAccuracy.csv"
            print fileName
            csvData = np.loadtxt(open(fileName, "rb"), delimiter=",", skiprows=0)
            csvData = [[int(x), np.nanmean([b[1] for b in csvData if b[0] == x])] for x in np.unique(csvData[:, :1])]
            csvData = np.array(csvData)
            samples = csvData[:, 0]
            accuracy = csvData[:, 1]
            print accuracy
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

        plt.legend(loc=4)

        plt.xticks(np.arange(xlimMin-1, xlim+5, 5.0))
        plt.yticks(np.arange(ylimMin-1, ylim+5, 5.0))
        plt.xlim(xlimMin-1, xlim+5)
        #plt.ylim(ylimMin-1, ylim)
        plt.ylim(0, ylim+5)
        plt.grid()
        #plt.show()
        if which == 0:
            plt.savefig("Desbalanced/analysis/"+desc+"_"+met+"_TPR.png")
        else:
            plt.savefig("Desbalanced/analysis/"+desc+"_"+met+"_BalancedAccuracy.png")
        #plt.savefig(desc+"_"+met+'.png')
