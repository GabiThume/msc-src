#ifndef _CLASSIFIER_H
#define _CLASSIFIER_H

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

#include "utils/dataStructure.h"

class Classifier{

    std::vector<double> accuracy, balancedAccuracy, precision, recall;
    int totalTest, totalTrain;
    std::string outputName;

    public:

      void bayesClassifier(cv::Mat dataTraining, cv::Mat labelsTraining, cv::Mat dataTesting, cv::Mat& result);
      void knn(cv::Mat dataTraining, cv::Mat labelsTraining, cv::Mat dataTesting, int k, cv::Mat& result);
      double calculateMean(std::vector<double> accuracy);
      double calculateStandardDeviation(std::vector<double> accuracy);
      void printAccuracy(double id, std::vector<std::vector<double> > fScore);
      double calculateBalancedAccuracy(cv::Mat confusionMat);
      std::vector<double> calculateFscore(cv::Mat confusionMat);
      cv::Mat confusionMatrix(int numClasses, cv::Mat labelsTesting, cv::Mat result, int print);
      std::vector< std::vector<double> > classify(double trainingRatio,	int numRepetition, Data data, std::string name, double id);
};

#endif
