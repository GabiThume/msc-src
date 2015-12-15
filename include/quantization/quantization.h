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

#ifndef QUANTIZATION_H_
#define QUANTIZATION_H_

#include <iostream>
#include <sstream>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

class GrayscaleConversion {

  string quantization[7] = {"Intensity", "Luminance", "Gleam", "MSB", "MSBModified", "BGR", "HSV"};
  public:
    int numColors;
    string getName(int method) {
      string name = ((sizeof(quantization)/sizeof(quantization[0])) > method) ? quantization[method] : "";
      return name;
    }

    GrayscaleConversion(int colors);
    void convert(int method, Mat img, Mat *gray);
    void Intensity(Mat I, Mat *Q, int num_colors);
    void Gleam(Mat I, Mat *Q, int num_colors);
    void Luminance(Mat I, Mat *Q, int num_colors);
    void Luma(Mat I, Mat *Q, int num_colors);
    void MSB(Mat I, Mat *Q, int num_colors);
    void MSBModified(Mat I, Mat *Q, int num_colors);

    void PlotHistogram(Mat hist);
    void correctGamma(Mat *I, double gamma);
    void reduceImageColors(Mat *img, int nColors);

};

void GrayscaleConversion::GrayscaleConversion(int colors) {
  numColors = colors;
}

#endif
