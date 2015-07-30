#ifndef _CLASSIFIER_H
#define _CLASSIFIER_H

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

#include "funcoesArquivo.h"

using namespace cv;
using namespace std;


class Classifier{

    vector<float> accuracy, balancedAccuracy, precision, recall, fScore;
    int totalTest, totalTrain, numClasses;
    pair<int, int> minority;
    string outputName;

    public:

    	void bayesClassifier(Mat, Mat, Mat, Mat&);
		void knn(Mat, Mat, Mat, Mat&);
        void classify(float, int, vector<Classes>, string);
        void findSmallerClass(Mat, int, int*, int*, int*);
        void printAccuracy();
};

#endif
