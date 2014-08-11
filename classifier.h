#ifndef _CLASSIFIER_H
#define _CLASSIFIER_H

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

using namespace cv;
using namespace std;

class Classifier{

    public:
        void bayes(Mat, Mat, int, float, int);
        void printAccuracy(vector<float>, int, int, int, int);
};

#endif
