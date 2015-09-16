/**
 * Auxiliar Functions for Feature Extraction:
 *	Image enhancement
 *	Image quantization
 *	Normalization
 * 	Distance Functions
 *
 *	Authors: Gabriela Thumé, Moacir Ponti, Luciana Escobar
 *	Universidade de São Paulo / ICMC / 2012-2015
 **/

#include "utils/funcoesAux.h"


/*******************************************************************************
    Gamma correction (power transformation)

        Controls the overall brightness of an image:
            gammaCorrectedImage = image ^ (1/crt_gamma)

    Requires:
    - img : image to be processed
    - gamma : parameter for apply gamma correction
*******************************************************************************/
void PlotHistogram(Mat hist) {
  int bins = hist.rows, j;
  normalize(hist, hist, 0, bins, NORM_MINMAX, -1, Mat());

  Mat hist_img(bins, bins, CV_8UC1, Scalar(0,0,0));
  for (j = 1; j < bins; j++) {
    line(hist_img,
      Point((j-1), bins - cvRound(hist.at<float>(j-1))),
      Point(j, bins - cvRound(hist.at<float>(j))),
      Scalar(255, 255, 255), 1, 8, 0);
  }

  namedWindow("histogram", CV_WINDOW_AUTOSIZE );
  imshow("histogram", hist_img);
  waitKey(0);
}

/*******************************************************************************
    Gamma correction (power transformation)

        Controls the overall brightness of an image:
            gammaCorrectedImage = image ^ (1/crt_gamma)

    Requires:
    - img : image to be processed
    - gamma : parameter for apply gamma correction
*******************************************************************************/
void correctGamma(Mat *I, double gamma) {

    Mat result, lut_matrix(1, 256, CV_8UC1);
    uchar *ptr = lut_matrix.ptr();
    double iGamma = 1.0 / gamma;

    if ((*I).channels() == 1){
        // Create a lookup table with a grayscale color ajusted for each color
        for( int i = 0; i < 256; i++ ){
            ptr[i] = (uchar)(pow((double)i/255.0, iGamma) * 255.0);
        }
        // Apply the correction in lutMatrix on input img
        LUT((*I), lut_matrix, (*I));
    }
}

/*******************************************************************************
    Reduce the number of colors in a single channel image

    Requires:
    - I: input image to be modified
    - nColors: final number of colors
*******************************************************************************/
void reduceImageColors(Mat *I, int nColors) {

	double min, max, stretch;
	Point maxLoc, minLoc;

  nColors = (nColors > 256) ? 256 : nColors;

  if ((*I).channels() == 1){
    minMaxLoc(*I, &min, &max, &minLoc, &maxLoc);
    stretch = ((double)((nColors -1)) / (max - min));
    (*I) = (*I) - min;
    (*I) = (*I) * stretch;
  }
  namedWindow( "Display window", WINDOW_AUTOSIZE );
  imshow("ReducedColors", *I);
  waitKey(0);
}

/* Remove null columns in feature space
  (under construction)
*/
void RemoveNullColumns(Mat *Feat) {

    int width = (*Feat).size().width;
    vector<int> marktoremove(width);

    for (int i = 0; i < width; i++) {
        Mat columni = (*Feat).col(i);
        double sumi = sum(columni)[0];
        if (sumi == 0)
            marktoremove[i] = 1;
        cout << "Sum " << i << " : " << sumi << endl;
    }
}

/* Histogram Normalization
   Requires:
	- hist: histogram to be normalized
	- histnorm: allocated histogram to store the result
	- vector size
	- normalization factor (1 for unity sum, > 1 for maximum*factor)
*/
void NormalizeHist(vector<int> *hist, float *histnorm, int size, int factor){

	int i;
	long int sum = 0, max = (*hist)[0];
	float e = 0.01;

	for (i = 0; i < size ; i++){
		sum += (*hist)[i];
		max = ((*hist)[i] > max) ? (*hist)[i] : max;
	}

	// if factor == 1 then vector with unity sum
	if (factor == 1){
		for (i = 0; i < size ; i++){
			histnorm[i] = (*hist)[i]/((float)sum+e);
		}
	}
	// if factor > 1 then vector with maximum value == factor
	else if (factor > 1){
		for (i = 0; i < size ; i++){
			histnorm[i] = ((*hist)[i]/(float)max)*(float)factor;
		}
	}
}

/* Distance Function Manhattan (l1-norm)
  Require:
 	- two histograms 'p' and 'q' to be compared
 	- size: histogram size
  Retorns:
 	- distance between 'p' and 'q'
*/
double distManhattan(double *p, double *q, int size){

	int i;
	double dist = 0;

	for (i = 0; i < size; i++){
		dist += fabs(p[i]-q[i]);
	}

	return dist;
}

/* Distance Function Euclidian (l2-norm)
  Require:
 	- two histograms 'p' and 'q' to be compared
 	- size: histogram size
  Retorns:
 	- distance between 'p' and 'q'
*/
double distEuclid(double *q, double *p, int size){

	int i;
	double dist = 0;

	for(i = 0; i < size; i ++){
		dist = dist + pow((q[i]-p[i]), 2);
	}

	dist = sqrt(dist);
	return dist;
}

/* Distance Function Chessboard (l_\infty-norm)
  Require:
 	- two histograms 'p' and 'q' to be compared
 	- size: histogram size
  Retorns:
 	- distance between 'p' and 'q'
*/
double distChessboard(double *p, double *q, int size){

	int i;
	double dist = 0;
	double maxVal = -1;

	for (i = 0; i < size; i++){
		dist = fabs(p[i]-q[i]);
		if (maxVal < dist)
		    maxVal = dist;
	}
	return maxVal;
}
