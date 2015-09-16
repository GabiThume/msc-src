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

void PlotHistogram(Mat hist);

/*
  Gamma correction function
  Requires:
	- img : image to be processed
	- gamma : parameter for apply gamma correction
  Returns:
	- gamma-corrected image
*/
void correctGamma(Mat *I, double gamma);

/*
  Function to reduce the number of colors in a single channel image
  Requires:
	- I: input image to be modified
	- nColors: final number of colors
*/
void reduceImageColors(Mat *I, int nColors);


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
