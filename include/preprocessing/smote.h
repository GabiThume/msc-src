#ifndef _SMOTE_H
#define _SMOTE_H

#include <opencv2/ml/ml.hpp>

using namespace cv;
using namespace std;

void populate(Mat minority, Mat neighbors, Mat *synthetic, int *index, int amount_smote, int i, int nearest_neighbors);

void computeNeighbors(Mat minority, int nearest_neighbors, Mat *neighbors);

Mat smote(Mat minority, int amount_smote, int nearest_neighbors);

class SMOTE{

    public:
        Mat smote(Mat, int, int);
        void computeNeighbors(Mat, int, Mat*);
        void populate(Mat, Mat, Mat*, int*, int, int, int);
};

#endif
