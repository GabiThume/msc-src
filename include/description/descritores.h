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

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <math.h>
#include <opencv2/nonfree/nonfree.hpp>

using namespace cv;
using namespace std;

const string quantizationMethod[7] = {"Intensity", "Luminance", "Gleam", "MSB", "MSBModified", "BGR", "HSV"};

typedef struct {
    int i;
    int j;
    uchar color;
} Pixel;

class FeatureExtraction {

    string descriptors[9] = {"BIC", "GCH", "CCV", "Haralick6", "ACC", "LBP", "HOG", "Contour", "Fisher"};
  public:
    int numColors;
    int normalization;
    int ccvThreshold;
    double resizeFactor;
    vector<int> accDistances;

    FeatureExtraction(int, int);

    string getName(int method);

    void extract(int method, Mat img, Mat *features);

    void VerifyNeighborPixel(Mat img, int index_height, int index_width,
      uchar pixel_color, vector< vector<bool> > *visited, queue<Pixel> *pixels,
      int *size_region);
    void FindNeighbor(Mat img, vector< vector<bool> > *visited,
      queue<Pixel> *pixels, int *size_region);
    void CalculateCCV(Mat img, Mat *features);
    void CCV(Mat img, Mat *features);

    void CalculateGCH(Mat img, Mat *features);
    void GCH(Mat img, Mat *features);

    void CalculateBIC(Mat img, Mat *features);
    void BIC(Mat img, Mat *features);

    vector<int> NearestNeighborAngle(int row, int col, int distance, int angle);

    void CoocurrenceMatrix(Mat img,
      vector< vector<double> > *co_occurence, int distance, int angle);
    void Haralick6(vector< vector<double> > co_occurence,
      Mat *features);
    void CalculateHARALICK(Mat img, Mat *features);
    void HARALICK(Mat img, Mat *features);

    vector < vector<int> > ChessboardNeighbors(int row, int col,
      int distance);
    void CalculateACC(Mat I, Mat *features);
    void ACC(Mat img, Mat *features);

    vector<int> initUniform();
    void CalculateLBP(Mat img, Mat *features);
    void LBP(Mat img, Mat *features);

    void CalculateHOG(Mat img, Mat *features);
    void HOG(Mat img, Mat *features);

    void CalculateContour(Mat img, Mat *features);
    void contourExtraction(Mat img, Mat *features);

    void RemoveNullColumns(Mat *features);
    void ZScoreNormalization(Mat *features);
    void MaxMinNormalization(Mat *features);

};

FeatureExtraction::FeatureExtraction(int colors, int norm) {
  numColors = colors;
  normalization = norm;
}

#endif
