#ifndef _SMOTE_H
#define _SMOTE_H

#include <opencv2/ml/ml.hpp>
#include <iostream>


void populate(cv::Mat minority, cv::Mat neighbors, cv::Mat *synthetic, int *index, int amount_smote, int i, int nearest_neighbors);

void computeNeighbors(cv::Mat minority, int nearest_neighbors, cv::Mat *neighbors);

cv::Mat smote(cv::Mat minority, int amount_smote, int nearest_neighbors);

class SMOTE{

    public:
        cv::Mat smote(cv::Mat, int, int);
        void computeNeighbors(cv::Mat, int, cv::Mat*);
        void populate(cv::Mat, cv::Mat, cv::Mat*, int*, int, int, int);
};

#endif
