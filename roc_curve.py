from matplotlib import pyplot as plt
import numpy as np

roc = ["Desbalanced/analysis/original_accuracy_", "Desbalanced/analysis/smote_accuracy_", "Desbalanced/analysis/rebalance_accuracy_"]

# descriptors = {"BIC", "GCH", "CCV", "Haralick6"}
# methods = {"Intensity", "Gleam", "Luminance", "MSB"}

descriptors = {"Haralick6"}
methods = {"MSB"}

for desc in descriptors:
    for met in methods:
        plt.clf()
        plt.title('Receiver operating characteristic (ROC)')
        # plt.title(desc+"_"+met)
        color = ["r", "g", "b"]
        labels = ["Original", "SMOTE", "Artificial Images"]
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        xlim, ylim, xlimMin, ylimMin = 0, 0, 999, 999

        for i in range(0, 3):
            fileName = roc[i]+desc+"_"+met+"_ROC.csv"
            print fileName
            csvData = np.loadtxt(open(fileName, "rb"), delimiter=",", skiprows=0)
            # csvData = [[int(x), np.nanmean([b[1] for b in csvData if b[0] == x])] for x in np.unique(csvData[:, :1])]
            # csvData = np.array(csvData)
            fpr = csvData[:, 0]
            tpr = csvData[:, 1]
            line1, = plt.plot(fpr, tpr, color[i]+'o', label=labels[i], linewidth=3)
    # plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % trapz(tpr, fpr))
            plt.plot(fpr, tpr, color[i]+'-', linewidth=3)
            if max(fpr) > xlim:
                xlim = np.ceil(max(fpr))
            if max(tpr) > ylim:
                ylim = np.ceil(max(tpr))
            if min(fpr) < xlimMin:
                xlimMin = np.round(min(fpr))
            if min(tpr) < ylimMin:
                ylimMin = np.round(min(tpr))

        plt.legend(loc=4)

        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0.0, 1.1, 0.1))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0, 1.0])
        plt.ylim([0.0, 1.0])
        # plt.xlim(xlimMin-1, xlim)
        # plt.ylim(ylimMin-1, ylim)
        plt.grid()
        #plt.show()
        plt.savefig('Desbalanced/analysis/ROC.png')
        #plt.savefig(desc+"_"+met+'.png')
