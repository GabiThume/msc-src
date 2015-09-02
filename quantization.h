#ifndef _QUANTIZATION_H
#define _QUANTIZATION_H

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <sstream>
#include <vector>
#include "funcoesAux.h"

using namespace cv;
using namespace std;

void QuantizationIntensity(Mat I, Mat *Q, int nColors);
void QuantizationGleam(Mat I, Mat *Q, int nColors);
void QuantizationLuminance(Mat I, Mat *Q, int nColors);
void QuantizationLuma(Mat I, Mat *Q, int nColors);
void QuantizationMSB(Mat I, Mat *Q, int nColors);

#endif
