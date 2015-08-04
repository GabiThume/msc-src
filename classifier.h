#ifndef _CLASSIFIER_H
#define _CLASSIFIER_H

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

#include "funcoesArquivo.h"

using namespace cv;
using namespace std;


class Classifier{

    vector<double> accuracy, balancedAccuracy, precision, recall, fScore;
    int totalTest, totalTrain, numClasses;
    string outputName;

    public:

    	void bayesClassifier(Mat, Mat, Mat, Mat&);
		void knn(Mat, Mat, Mat, Mat&);
        void classify(double, int, vector<Classes>, string, int);
        int findSmallerClass(vector<Classes>);
        void printAccuracy(int id);
};

#endif
