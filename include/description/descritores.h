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
#include "utils/funcoesAux.h"
#include <opencv2/nonfree/nonfree.hpp>

// #include "vlfeat/vl/sift.h"
#include <vl/sift.h>

using namespace cv;
using namespace std;

const string quantizationMethod[4] = {"Intensity", "Luminance", "Gleam", "MSB"};
const string descriptorMethod[9] = {"BIC", "GCH", "CCV", "Haralick6", "ACC", "LBP", "HOG", "Contour", "Fisher"};

typedef struct {
    int i;
    int j;
    uchar color;
} Pixel;

void find_neighbor(Mat img, queue<Pixel> *pixels, int *visited, long int *tam_reg);
void CCV(Mat img, Mat *features, int nColor, int oNorm, int threshold);
void GCH(Mat I, Mat *features, int nColor, int oNorm);
void BIC(Mat I, Mat *features, int nColor, int oNorm);
void CoocurrenceMatrix(Mat img, vector< vector<double> > *co_occurence,
                      int colors, int distance, int angle);
void Haralick6(vector< vector<double> > co_occurence, Mat *features);
void CalculateHARALICK(Mat img, Mat *features, int colors, int normalization);
void HARALICK(Mat img, Mat *features, int colors, int normalization);
void ACC(Mat I, Mat *features, int colors, int normalization, vector<int> distances);
void LBP(Mat img, Mat *features, int colors, int normalization);
void HOG(Mat img, Mat *features, int colors, int normalization);
void contourExtraction(Mat Img, Mat *features, int colors, int normalization);

#endif
