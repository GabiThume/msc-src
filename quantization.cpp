/**
*	Authors: Gabriela Thumé, Moacir Ponti, Luciana Escobar
*	Universidade de São Paulo / ICMC / 2012-2015
**/

#include "quantization.h"

/*******************************************************************************
Image quantization by Gamma corrected Intensity

GrayscaleImage = (R + G + B)*1/3

Requires:
- I: image to be converted into grayscale
- Q: image to store quantized version
- nColors: number of colors after quantization
*******************************************************************************/
void QuantizationIntensity(Mat I, Mat *Q, int nColors){

    vector<Mat> imColors(3);
    (*Q).create((*Q).size(), CV_8UC1);

    // Split input image into separated channels
    split(I, imColors);

    // Compute grayscale image
    (*Q) = (imColors[0] + imColors[1] + imColors[2])/3.0;

    // Gamma correction after quantization
    correctGamma(Q, 2.2);

    // Reduce number of colors if necessary
    if (nColors < 256){
        reduceImageColors(Q, nColors);
    }
}

/*******************************************************************************
Image quantization by Gleam

GrayscaleImage = (R' + G' + B')*1/3

Requires
- I: image to be converted
- Q: image to store quantized version
- nColors: number of colors after quantization
*******************************************************************************/
void QuantizationGleam(Mat I, Mat *Q, int nColors){

    vector<Mat> imColors(3);
    (*Q).create((*Q).size(), CV_8UC1);

    // using image 'out', split channels in tree matrix
    split(I, imColors);

    correctGamma(&(imColors[0]), 2.2);
    correctGamma(&(imColors[1]), 2.2);
    correctGamma(&(imColors[2]), 2.2);

    // Sum 1/3 of each channel using gamma corrected pixels
    (*Q) = (imColors[0] + imColors[1] + imColors[2])/3.0;

    // Reduce number of colors if necessary
    if (nColors < 256){
        reduceImageColors(Q, nColors);
    }
}

/*******************************************************************************
Image quantization by Gamma corrected Luminance

Designed to match human brightness perception by using a weighted
combination of the RGB channels:
GrayscaleImage = 0.299*R + 0.587*G + 0.114*B

Requires
- I: image to be converted
- Q: image to store quantized version
- nColor : number of colors after quantization
*******************************************************************************/
void QuantizationLuminance(Mat I, Mat *Q, int nColors){

    vector<Mat> imColors(3);
    (*Q).create((*Q).size(), CV_8UC1);

    // split image in RGB channels
    split(I, imColors);

    // Weighted combination considering input image is BGR
    (*Q) = (0.299*imColors[2]) + (0.587*imColors[1]) + (0.114*imColors[0]);

    // Gamma correction
    correctGamma(Q, 2.2);

    // Reduce number of colors if necessary
    if (nColors < 256) {
        reduceImageColors(Q,nColors);
    }
}

/*******************************************************************************
Image quantization by Gamma corrected Luma

Used in high-definition televisions:
GrayscaleImage = 0.2126*R' + 0.7152*G' + 0.0722*B'

Requires
- I: image to be converted
- Q: image to store quantized version
- nColor : number of colors after quantization
*******************************************************************************/
void QuantizationLuma(Mat I, Mat *Q, int nColors){

    vector<Mat> imColors(3);
    (*Q).create((*Q).size(), CV_8UC1);

    // split image in RGB channels
    split(I, imColors);

    correctGamma(&(imColors[0]), 2.2);
    correctGamma(&(imColors[1]), 2.2);
    correctGamma(&(imColors[2]), 2.2);

    // Weighted combination considering input image is BGR
    (*Q) = (0.2126*imColors[2]) + (0.7152*imColors[1]) + (0.0722*imColors[0]);

    // Reduce number of colors if necessary
    if (nColors < 256){
        reduceImageColors(Q,nColors);
    }
}

/*******************************************************************************
Image quantization using the Most Significant Bits (MSB)

Calculate the most significant bits of color channels, in order to
emphasize the chromatic differences. Based on the range of stimulus of
the rod cells for visible light wavelenghts. The order of preference is
G, R and B.

Require:
- I: image to be converted
- Q: image to store quantized version
- nColors: number of colors after quantization
*******************************************************************************/
void QuantizationMSB(Mat I, Mat *Q, int nColors){

    int bitsc, rest, k;
    MatIterator_<Vec3b> itI, endI;
    MatIterator_<uchar> itQ, endQ;
    (*Q).create((*Q).size(), CV_8UC1);

    // Compute amount of bits needed to obtain nColors
    bitsc = log(nColors)/log(2);
    // Compute amount of bits used from channel
    int GRBbits[3] = {bitsc/3, bitsc/3, bitsc/3};

    // Check if there are bits left after equal division
    k = 0;
    for (rest = bitsc % 3; rest > 0; rest--){
        GRBbits[k]++;
        k = (k+1) % 3;
    }

    for(itQ = (*Q).begin<uchar>(), endQ = (*Q).end<uchar>(), itI = I.begin<Vec3b>(), endI = I.end<Vec3b>(); itI != endI; ++itI, ++itQ){

        uchar dG = (8-GRBbits[0]);
        uchar dR = (8-GRBbits[1]);
        uchar dB = (8-GRBbits[2]);

        // obtain mask for each channel
        uchar Ga = ((int)(pow(2,GRBbits[0]))-1) << dG;
        uchar Ra = ((int)(pow(2,GRBbits[1]))-1) << dR;
        uchar Ba = ((int)(pow(2,GRBbits[2]))-1) << dB;

        // Get pixels of individual channels in the input image
        uchar B = (*itI)[0];
        uchar G = (*itI)[1];
        uchar R = (*itI)[2];

        // generate codes for each channel
        uchar C1 = (B & Ba) >> dG;                  // extract MSBs from B
        uchar C2 = (R & Ra) >> (dG-GRBbits[1]);        // extract MSBs from R
        uchar C3 = (G & Ga) >> (dG-GRBbits[1]-GRBbits[2]);// extract MSBs from G

        // Merge the bit codes
        uchar newPixel = C1 | C2 | C3;

        // Store in the new image
        (*itQ) = newPixel;

        // %todo: o que fazer nesse caso?
        if (newPixel > 255)
        cout << "\nColor overflow: " << newPixel << endl;
    }

    // Reduce number of colors if necessary
    if (nColors < 256) {
        reduceImageColors(Q,nColors);
    }
}
