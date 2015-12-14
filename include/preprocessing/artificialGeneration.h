#ifndef _ARTIFICIAL_H
#define _ARTIFICIAL_H

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <dirent.h>
#include <iostream>
#include <string>
#include <vector>

#include <cstring>
#include <cstdio>

#include "../lib/Saliency/GMRsaliency.h"
#include "utils/funcoesArquivo.h"
#include "classification/classifier.h"

using namespace cv;
using namespace std;

class Artificial{

  public:
    string generate(string base, string newDirectory, int whichOperation);
    void GenerateImage(vector<Mat> images, string name, int total, int generationType);
    Mat generateBlur(Mat originalImage, int blurType);
    Mat generateNoise(Mat img);
    Mat generateBlending(Mat first, Mat second);
    Mat generateUnsharp(Mat originalImage);
    Mat generateComposition(Mat originalImage, vector<Mat> images, int total,
      int fator, bool option);
    Mat generateThreshold(Mat first, Mat second);
    Mat generateSaliency(Mat first, Mat second);
    Mat generateSmoteImg(Mat first, Mat second);
    vector<Classes> generateImagesFromData(vector<Classes> original_data,
			string newDirectory,
			int whichOperation);
};

#endif
