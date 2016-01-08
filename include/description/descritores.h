/*
Copyright (c) 2015, All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the
distribution.
    * Neither the name of Gabriela Thumé nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors:  Gabriela Thumé (gabithume@gmail.com)
          Moacir Antonelli Ponti (moacirponti@gmail.com)
Universidade de São Paulo / ICMC
Master's thesis in Computer Science
*/

#ifndef _DESCRITORES_H
#define _DESCRITORES_H

#include <queue>
#include <vector>
#include <math.h>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

typedef struct {
    int i;
    int j;
    uchar color;
} Pixel;

class FeatureExtraction {

    std::string descriptors[9] = {"BIC", "GCH", "CCV", "Haralick6", "ACC", "LBP", "HOG", "Contour", "Fisher"};
  public:
    int method;
    int numColors;
    int normalization;
    int ccvThreshold;
    double resizeFactor;
    std::vector<int> accDistances;

    std::string getName(void);

    void extract(int colors, int norm, int method, cv::Mat img, cv::Mat *features);

    void VerifyNeighborPixel(cv::Mat img, int index_height, int index_width,
      uchar pixel_color, std::vector< std::vector<bool> > *visited, std::queue<Pixel> *pixels,
      int *size_region);
    void FindNeighbor(cv::Mat img, std::vector< std::vector<bool> > *visited,
      std::queue<Pixel> *pixels, int *size_region);
    void CalculateCCV(cv::Mat img, cv::Mat *features);
    void CCV(cv::Mat img, cv::Mat *features);

    void CalculateGCH(cv::Mat img, cv::Mat *features);
    void GCH(cv::Mat img, cv::Mat *features);

    void CalculateBIC(cv::Mat img, cv::Mat *features);
    void BIC(cv::Mat img, cv::Mat *features);

    std::vector<int> NearestNeighborAngle(int row, int col, int distance, int angle);

    void CoocurrenceMatrix(cv::Mat img,
      std::vector< std::vector<double> > *co_occurence, int distance, int angle);
    void Haralick6(std::vector< std::vector<double> > co_occurence,
      cv::Mat *features);
    void CalculateHARALICK(cv::Mat img, cv::Mat *features);
    void HARALICK(cv::Mat img, cv::Mat *features);

    std::vector < std::vector<int> > ChessboardNeighbors(int row, int col,
      int distance);
    void CalculateACC(cv::Mat I, cv::Mat *features);
    void ACC(cv::Mat img, cv::Mat *features);

    std::vector<int> initUniform();
    void CalculateLBP(cv::Mat img, cv::Mat *features);
    void LBP(cv::Mat img, cv::Mat *features);

    void CalculateHOG(cv::Mat img, cv::Mat *features);
    void HOG(cv::Mat img, cv::Mat *features);

    void CalculateContour(cv::Mat img, cv::Mat *features);
    void contourExtraction(cv::Mat img, cv::Mat *features);

    void RemoveNullColumns(cv::Mat *features);
    void ZScoreNormalization(cv::Mat *features);
    void MaxMinNormalization(cv::Mat *features, int norm);
    void reduceImageColors(cv::Mat *img, int nColors);

};

#endif
