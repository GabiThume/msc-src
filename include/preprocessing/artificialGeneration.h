#ifndef _ARTIFICIAL_H
#define _ARTIFICIAL_H

#include <fstream>
#include <dirent.h>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../lib/Saliency/GMRsaliency.h"
#include "classification/classifier.h"


class Artificial{

  public:
    std::string generate(std::string base, std::string newDirectory, int whichOperation);
    void GenerateImage(std::vector<cv::Mat> images, std::string name, int total, int generationType);
    cv::Mat generateBlur(cv::Mat originalImage, int blurType);
    cv::Mat generateNoise(cv::Mat img);
    cv::Mat generateBlending(cv::Mat first, cv::Mat second);
    cv::Mat generateUnsharp(cv::Mat originalImage);
    cv::Mat generateComposition(cv::Mat originalImage, std::vector<cv::Mat> images, int total,
      int fator, bool option);
    cv::Mat generateThreshold(cv::Mat first, cv::Mat second);
    cv::Mat generateSaliency(cv::Mat first, cv::Mat second);
    cv::Mat generateSmoteImg(cv::Mat first, cv::Mat second);
    std::vector<ImageClass> generateImagesFromData(std::vector<ImageClass> original_data,
			std::string newDirectory,
			int whichOperation);
};

#endif
