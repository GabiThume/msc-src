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

#include "Saliency/GMRsaliency.h"
#include "funcoesArquivo.h"
#include "classifier.h"

using namespace cv;
using namespace std;

class Artificial{

    public:
        void generate(string base, int whichOperation);
        Mat generateBlur(Mat originalImage);
        Mat generateNoise(Mat img);
        Mat generateBlending(Mat originalImage, vector<Mat> images, int total);
        Mat generateUnsharp(Mat originalImage);
		Mat generateComposition(Mat originalImage, vector<Mat> images, int total, int fator);
		Mat generateThreshold(Mat originalImage, vector<Mat> images, int total);
		Mat generateSaliency(Mat originalImage, vector<Mat> images, int total);
};


#endif
