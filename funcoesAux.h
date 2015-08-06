/**
 * Auxiliar Functions for Feature Extraction:
 *	Image enhancement
 *	Image quantization
 *	Normalization
 * 	Distance Functions
 *
 *	Authors: Moacir Ponti, Luciana Escobar
 *	Universidade de São Paulo / ICMC / 2012-2014
 **/
#ifndef _FUNCOESAUX_H
#define _FUNCOESAUX_H

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <sstream>
#include <vector>

using namespace cv;
using namespace std;

/*
  Gamma correction function
  Requires:
	- img : image to be processed
	- gamma : parameter for apply gamma correction
  Returns:
	- gamma-corrected image
*/
Mat correctGamma(Mat *img, double gamma );

/*
  Function to reduce the number of colors in a single channel image
  Requires:
	- I: input image to be modified
	- nColors: final number of colors
*/
void reduceImageColors(Mat *I, int nColors);

/*
  Image quantization by Gamma corrected Intensity
  Requires
	- I : image to be converted
	- Q : image to store quantized version
	- nColors : number of colors after quantization
*/
void QuantizationIntensity(Mat *I, Mat *Q, int nColors);


/*
  Image quantization by Gleam
   Requires
	- I : image to be converted
	- Q : image to store quantized version
	- nColors : number of colors after quantization
*/
void QuantizationGleam(Mat *I, Mat *Q, int nColors);


/*
  Image quantization by gamma-corrected Luminance
   Requires
	- I : image to be converted
	- Q : image to store quantized version
	- nColors : number of colors after quantization
*/
void QuantizationLuminance(Mat *I, Mat *Q, int nColors);


/*
  Image quantization using the Most Significant Bits (MSB)
  Require:
	- I : image to be converted
	- Q : image to store quantized version
	- nColors : number of colors after quantization
*/
void QuantizationMSB(Mat *I, Mat *Q, int nColors);


/*
  Remove null columns in feature space
  (under construction)

*/
void RemoveNullColumns(Mat *Feat);


/*
   Histogram Normalization
   Requires:
	- hist: histogram to be normalized
	- histnorm: allocated histogram to store the result
	- vector size
	- normalization factor (1 for unity sum, > 1 for maximum*factor)
*/
void NormalizeHist(vector<int> *hist, float *histnorm, int size, int factor);


/*
  Distance Function Manhattan (l1-norm)
  Require:
 	- two histograms 'p' and 'q' to be compared
 	- size: histogram size
  Retorns:
 	- distance between 'p' and 'q'
*/
double distManhattan(double *p, double *q, int size);


/*
  Distance Function Euclidian (l2-norm)
  Require:
 	- two histograms 'p' and 'q' to be compared
 	- size: histogram size
  Retorns:
 	- distance between 'p' and 'q'
*/
double distEuclid(double *q, double *p, int size);


/*
  Distance Function Chessboard (l_\infty-norm)
  Require:
 	- two histograms 'p' and 'q' to be compared
 	- size: histogram size
  Retorns:
 	- distance between 'p' and 'q'
*/
double distChessboard(double *p, double *q, int size);


#endif
