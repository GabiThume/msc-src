#ifndef _CLASSIFIER_H
#define _CLASSIFIER_H

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

using namespace cv;
using namespace std;

class Classifier{

    vector<float> accuracy, balancedAccuracy, precision, recall, fScore;
    int totalTest, totalTrain, numClasses;
    pair<int, int> minority;
    string outputName;

    public:
        void bayes(float, int, Mat, Mat, int, pair<int, int>, string);
        void findSmallerClass(Mat, int, int&, int&, int&);
        void printAccuracy();
};

#endif
