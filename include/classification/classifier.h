#ifndef _CLASSIFIER_H
#define _CLASSIFIER_H

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

#include "utils/funcoesArquivo.h"

class Classifier{

    vector<double> accuracy, balancedAccuracy, precision, recall;
    int totalTest, totalTrain, numClasses;
    string outputName;

    public:

      void bayesClassifier(Mat, Mat, Mat, Mat&);
      void knn(Mat, Mat, Mat, Mat&);
      vector<vector<double> > classify(double, int, vector<Classes>, string, double);
      int findSmallerClass(vector<Classes>, int*);
      void printAccuracy(double, vector<vector<double> >);
      double calculateMean(vector<double>);
      double calculateStandardDeviation(vector<double>);
};

#endif
