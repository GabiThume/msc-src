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

/* 
  Gamma correction function
  Requires:
	- img : image to be processed
	- gamma : parameter for apply gamma correction
  Returns:
	- gamma-corrected image
*/
Mat correctGamma( Mat& img, double gamma ) {
    //double inverse_gamma = 1.0 / gamma;
    
    Mat lut_matrix(1, 256, CV_8UC1 );
    uchar *ptr = lut_matrix.ptr();
    for( int i = 0; i < 256; i++ ) {
	  ptr[i] = (int)( pow( (double) i / 255.0, gamma ) * 255.0 );
    }
    
    Mat result;
    LUT( img, lut_matrix, result );
    
    return result;
}

/*
  Function to reduce the number of colors in a single channel image
  Requires:
	- I: input image to be modified
	- nColors: final number of colors
*/
void reduceImageColors(Mat &I, int nColors) {

	if (nColors >= 256) return;

	double min, max;
	Point maxLoc, minLoc;
      	minMaxLoc(I, &min, &max, &minLoc, &maxLoc);
      	double stretch = ((double)((nColors-1)) / (max - min ));
      	I = I - min;
      	I = I * stretch;
}


/*
  Image quantization by Gamma corrected Intensity 
  Requires
	- I : image to be converted
	- Q : image to store quantized version
	- nColors : number of colors after quantization
*/
void QuantizationIntensity(Mat &I, Mat &Q, int nColors) 
{
    vector<Mat> imColors(3);

    // separate input image into RGB channels
    split(I, imColors);

    // Compute Intensity image
    Q = (imColors[0]/3) + (imColors[1]/3) + (imColors[2]/3);
    // Gamma function
    Q = correctGamma(Q, 1/2.2);

    // reduce number of colors
    if (nColors < 256) reduceImageColors(Q,nColors);
}


/*
  Image quantization by Gleam
   Requires
	- I : image to be converted
	- Q : image to store quantized version
	- nColors : number of colors after quantization
*/
void QuantizationGleam(Mat &I, Mat &Q, int nColors) 
{
    //cout << " >>> Gleam\n";
    vector<Mat> imColors(3);
    //potencia usada para a correção gamma dos canais
    double pot = 1/2.2;

    // a partir da imagem 'out' (ja convertida), separa os canais nas
    // tres matrizes definidas em imColors
    split(I, imColors);

    imColors[0] = correctGamma(imColors[0], 1/2.2);
    imColors[1] = correctGamma(imColors[1], 1/2.2);
    imColors[2] = correctGamma(imColors[2], 1/2.2);
    
    //somatorio de 1/3 de cada canal com a correção gamma em cada pixel
    Q = imColors[0]/(3.0) + imColors[1]/(3.0) + imColors[2]/(3.0);

    // reduce number of colors
    if (nColors < 256) reduceImageColors(Q,nColors);
}


/*
  Image quantization by gamma-corrected Luminance
   Requires
	- I : image to be converted
	- Q : image to store quantized version
	- nColors : number of colors after quantization
*/
void QuantizationLuminance(Mat &I, Mat &Q, int nColors) 
{
    vector<Mat> imColors(3);

    // split image in RGB channels
    split(I, imColors);

    // Luminance image computation
    Q = (0.299*imColors[2]) + (0.587*imColors[1]) + (0.114*imColors[0]);
    // Gamma correction
    Q = correctGamma(Q, 1/2.2);

    // reduce number of colors
    if (nColors < 256) reduceImageColors(Q,nColors);
}


/*
  Image quantization using the Most Significant Bits (MSB)
  Require:
	- I : image to be converted
	- Q : image to store quantized version
	- nColors : number of colors after quantization
*/
void QuantizationMSB(Mat &I, Mat &Q, int nColors) 
{
		// get image size
	Size imgSize = I.size();
		
		// computes number of bits needed to obtain nColors
	int bitsc = log(nColors)/log(2); 

	// computers number of bits per channel
	int cc = (int)(bitsc/3);
	int GRBb[3]={cc,cc,cc}; 
	
	// check if there are bits left after equal division
	int rest = (bitsc % 3); 
	int k;
	for (k = 0 ;rest > 0; rest--, k = (k+1)%3) {
		GRBb[k]++;
	}
	
	MatIterator_<Vec3b> it, end;
	MatIterator_<uchar> it2, end2;
	for( it2 = Q.begin<uchar>(), end2 = Q.end<uchar>(), it = I.begin<Vec3b>(), end = I.end<Vec3b>(); it != end; ++it, ++it2)
	{
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
		if (newcolor > 255) cout << "color overflow: " << newcolor << endl;
	}	

}

/* 
  Remove null columns in feature space
  (under construction)

*/
void RemoveNullColumns(Mat &Feat) {
      int height = Feat.size().height;
      int width = Feat.size().width;
      
      vector<int> marktoremove(width);
      for (int i = 0; i < width; i++) {
	  Mat columni = Feat.col(i);
	  double sumi = sum(columni)[0];
	  if (sumi == 0) marktoremove[i] = 1;
	  cout << "Sum " << i << " : " << sumi << endl;
      }
}

/* 
   Histogram Normalization
   Requires:
	- hist: histogram to be normalized
	- histnorm: allocated histogram to store the result
	- vector size
	- normalization factor (1 for unity sum, > 1 for maximum*factor)
*/
void NormalizeHist(long int *hist, float *histnorm, int size, int factor) 
{
	int i;
	long int sum = 0;
	long int max = hist[0];
	float e = 0.01;
	
	for (i = 0; i < size ; i++)
	{
		sum += hist[i];
		max = (hist[i] > max) ? hist[i] : max;
	}
	
	// factor == 1 -> vector with unity sum
	if (factor == 1)
	{
		for (i = 0; i < size ; i++) 
		{
			histnorm[i] = hist[i]/((float)sum+e);
		}
	} 

	// factor > 1 -> vector with maximum value == factor
	else if (factor > 1) 
	{
		for (i = 0; i < size ; i++) 
		{
			histnorm[i] = (hist[i]/(float)max)*(float)factor;
		}
	}
}


/*	
  Distance Function Manhattan (l1-norm)
  Require:
 	- two histograms 'p' and 'q' to be compared
 	- size: histogram size
  Retorns:
 	- distance between 'p' and 'q'
*/
double distManhattan(double *p, double *q, int size) 
{
	int i;
	double dist = 0;
	
	for (i = 0; i < size; i++) 
	{
		dist += fabs(p[i]-q[i]);      
	}
	
	return dist;  
}


/*	
  Distance Function Euclidian (l2-norm)
  Require:
 	- two histograms 'p' and 'q' to be compared
 	- size: histogram size
  Retorns:
 	- distance between 'p' and 'q'
*/
double distEuclid(double *q, double *p, int size)
{
	int i;
	double dist = 0;
	
	for(i = 0; i < size; i ++) 
	{
		dist = dist + pow((q[i]-p[i]),2);
	}
	
	dist = sqrt(dist);
	
	return dist;
}


/*	
  Distance Function Chessboard (l_\infty-norm)
  Require:
 	- two histograms 'p' and 'q' to be compared
 	- size: histogram size
  Retorns:
 	- distance between 'p' and 'q'
*/
double distChessboard(double *p, double *q, int size)
{
	int i;
	double dist = 0;
	double maxVal = -1;
	
	for (i = 0; i < size; i++)
	{
		dist = fabs(p[i]-q[i]);
		if (maxVal < dist) { maxVal = dist; }
	}
	
	return maxVal;
}
