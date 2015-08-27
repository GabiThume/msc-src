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

#include "funcoesAux.h"

/* Gamma correction function
  Requires:
	- img : image to be processed
	- gamma : parameter for apply gamma correction
  Returns:
	- gamma-corrected image
*/
Mat correctGamma(Mat img, double gamma ) {

    Mat result;
    Mat lut_matrix(1, 256, CV_8UC1 );
    uchar *ptr = lut_matrix.ptr();
    //double inverse_gamma = 1.0 / gamma;

    for( int i = 0; i < 256; i++ ){
        ptr[i] = (int)(pow((double)i/255.0, gamma) * 255.0);
    }
    LUT(img, lut_matrix, result);

    return result;
}

/* Function to reduce the number of colors in a single channel image
  Requires:
	- I: input image to be modified
	- nColors: final number of colors
*/
void reduceImageColors(Mat *I, int nColors) {

	double min, max, stretch;
	Point maxLoc, minLoc;

	if (nColors >= 256) return;

    minMaxLoc(*I, &min, &max, &minLoc, &maxLoc);
    stretch = ((double)((nColors-1)) / (max - min));
    (*I) = (*I) - min;
    (*I) = (*I) * stretch;
    namedWindow( "Display window", WINDOW_AUTOSIZE );
    imshow("Reduced", *I);
    waitKey(0);
}

/* Image quantization by Gamma corrected Intensity
  Requires
	- I : image to be converted
	- Q : image to store quantized version
	- nColors : number of colors after quantization
*/
void QuantizationIntensity(Mat I, Mat *Q, int nColors){

    vector<Mat> imColors(3);
    (*Q).create((*Q).size(), CV_8U);
    // separate input image into RGB channels
    split(I, imColors);
    // Compute Intensity image
    (*Q) = (imColors[0]/3) + (imColors[1]/3) + (imColors[2]/3);
    // Gamma function
    (*Q) = correctGamma((*Q), 1/2.2);
    // reduce number of colors
    if (nColors < 256)
        reduceImageColors(Q,nColors);
}

/* Image quantization by Gleam
   Requires
	- I : image to be converted
	- Q : image to store quantized version
	- nColors : number of colors after quantization
*/
void QuantizationGleam(Mat I, Mat *Q, int nColors){

    vector<Mat> imColors(3);
    double pot = 1.0/2.2;
    (*Q).create((*Q).size(), CV_8U);

    // using image 'out', split channels in tree matrix
    split(I, imColors);

    imColors[0] = correctGamma(imColors[0], pot);
    imColors[1] = correctGamma(imColors[1], pot);
    imColors[2] = correctGamma(imColors[2], pot);

    // sum of 1/3 of each channel with gamma correction for each pixel
    (*Q) = imColors[0]/(3.0) + imColors[1]/(3.0) + imColors[2]/(3.0);

    // reduce number of colors
    if (nColors < 256)
        reduceImageColors(Q, nColors);
}

/* Image quantization by gamma-corrected Luminance
   Requires
	- I : image to be converted
	- Q : image to store quantized version
	- nColors : number of colors after quantization
*/
void QuantizationLuminance(Mat I, Mat *Q, int nColors){

    vector<Mat> imColors(3);
    (*Q).create((*Q).size(), CV_8U);
    // split image in RGB channels
    split(I, imColors);
    // Luminance image computation
    (*Q) = (0.299*imColors[2]) + (0.587*imColors[1]) + (0.114*imColors[0]);
    // Gamma correction
    (*Q) = correctGamma((*Q), 1/2.2);
    // reduce number of colors
    if (nColors < 256)
        reduceImageColors(Q,nColors);
}

/* Image quantization using the Most Significant Bits (MSB)
  Require:
	- I : image to be converted
	- Q : image to store quantized version
	- nColors : number of colors after quantization
*/
void QuantizationMSB(Mat I, Mat *Q, int nColors){

    int bitsc, cc, rest, k;
	MatIterator_<Vec3b> it, end;
	MatIterator_<uchar> it2, end2;
    (*Q).create((*Q).size(), CV_8U);

	// computes number of bits needed to obtain nColors
	bitsc = log(nColors)/log(2);
	// computers number of bits per channel
	cc = (int)(bitsc/3);
	int GRBb[3]={cc,cc,cc};

	// check if there are bits left after equal division
	rest = (bitsc % 3);
	for (k = 0; rest > 0; rest--, k = (k+1)%3) {
		GRBb[k]++;
	}

	for( it2 = (*Q).begin<uchar>(), end2 = (*Q).end<uchar>(), it = I.begin<Vec3b>(), end = I.end<Vec3b>(); it != end; ++it, ++it2){
		uchar dR = (8-GRBb[1]);
		uchar dG = (8-GRBb[0]);
		uchar dB = (8-GRBb[2]);
		// obtain mask for each channel
		uchar Ra = ((int)(pow(2,GRBb[1]))-1) << dR;
		uchar Ga = ((int)(pow(2,GRBb[0]))-1) << dG;
		uchar Ba = ((int)(pow(2,GRBb[2]))-1) << dB;

		// get pixels of individual channels
		uchar R = (*it)[2];
		uchar G = (*it)[1];
		uchar B = (*it)[0];

		// generate codes for each channel
		uchar C1 = (B & Ba) >> dG;                  // extract MSBs from G
		uchar C2 = (R & Ra) >> (dG-GRBb[1]);        // extract MSBs from R and move it
		uchar C3 = (G & Ga) >> (dG-GRBb[1]-GRBb[2]);// extract MSBs from B and move it

		// operator OR to merge the tree bit codes
		uchar newcolor = C1 | C2 | C3;

		// store in the Matrix
		(*it2) = newcolor;

		// check for overflow
		if (newcolor > 255)
		    cout << "\nColor overflow: " << newcolor << endl;
	}
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
