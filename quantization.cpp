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

#include <vector>
#include "quantization.h"

/*******************************************************************************
Image quantization by Gamma corrected Intensity

GrayscaleImage = (R + G + B)*1/3

Requires:
- I: image to be converted into grayscale
- Q: image to store quantized version
- num_colors: number of colors after quantization
*******************************************************************************/
void QuantizationIntensity(Mat I, Mat *Q, int num_colors) {
  vector<Mat> imColors(3);
  (*Q).create(I.size(), CV_8UC1);

  // Split input image into separated channels
  split(I, imColors);

  // Compute grayscale image
  (*Q) = (imColors[0] + imColors[1] + imColors[2])/3.0;

  // Gamma correction after quantization
  correctGamma(Q, 2.2);

  // Reduce number of colors if necessary
  if (num_colors < 256) {
    reduceImageColors(Q, num_colors);
  }
}

/*******************************************************************************
Image quantization by Gleam

GrayscaleImage = (R' + G' + B')*1/3

Requires
- I: image to be converted
- Q: image to store quantized version
- num_colors: number of colors after quantization
*******************************************************************************/
void QuantizationGleam(Mat I, Mat *Q, int num_colors) {
  vector<Mat> imColors(3);
  (*Q).create(I.size(), CV_8UC1);

  split(I, imColors);

  correctGamma(&(imColors[0]), 2.2);
  correctGamma(&(imColors[1]), 2.2);
  correctGamma(&(imColors[2]), 2.2);

  // Sum 1/3 of each channel using gamma corrected pixels
  (*Q) = (imColors[0] + imColors[1] + imColors[2])/3.0;

  // Reduce number of colors if necessary
  if (num_colors < 256) {
    reduceImageColors(Q, num_colors);
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
void QuantizationLuminance(Mat I, Mat *Q, int num_colors) {
  vector<Mat> imColors(3);
  (*Q).create(I.size(), CV_8UC1);

  // split image in RGB channels
  split(I, imColors);

  // Weighted combination considering input image is BGR
  (*Q) = (0.299*imColors[2]) + (0.587*imColors[1]) + (0.114*imColors[0]);

  // Gamma correction
  correctGamma(Q, 2.2);

  // Reduce number of colors if necessary
  if (num_colors < 256) {
    reduceImageColors(Q, num_colors);
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
void QuantizationLuma(Mat I, Mat *Q, int num_colors) {
  vector<Mat> imColors(3);
  (*Q).create(I.size(), CV_8UC1);

  // split image in RGB channels
  split(I, imColors);

  correctGamma(&(imColors[0]), 2.2);
  correctGamma(&(imColors[1]), 2.2);
  correctGamma(&(imColors[2]), 2.2);

  // Weighted combination considering input image is BGR
  (*Q) = (0.2126*imColors[2]) + (0.7152*imColors[1]) + (0.0722*imColors[0]);

  // Reduce number of colors if necessary
  if (num_colors < 256) {
    reduceImageColors(Q, num_colors);
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
- num_colors: number of colors after quantization
*******************************************************************************/
void QuantizationMSB(Mat I, Mat *Q, int num_colors) {
  int bitsc, rest, k;
  MatIterator_<Vec3b> itI, endI;
  MatIterator_<uchar> itQ, endQ;
  (*Q).create(I.size(), CV_8UC1);

  // Compute amount of bits needed to obtain num_colors
  bitsc = log(num_colors)/log(2);
  // Compute amount of bits used from channel
  int GRBbits[3] = {bitsc/3, bitsc/3, bitsc/3};

  // Check if there are bits left after equal division
  k = 0;
  for (rest = bitsc % 3; rest > 0; rest--) {
    GRBbits[k]++;
    k = (k+1) % 3;
  }

  itQ = (*Q).begin<uchar>();
  endQ = (*Q).end<uchar>();
  itI = I.begin<Vec3b>();
  endI = I.end<Vec3b>();
  for (; itI != endI; ++itI, ++itQ) {
    uchar dG = (8-GRBbits[0]);
    uchar dR = (8-GRBbits[1]);
    uchar dB = (8-GRBbits[2]);

    // obtain mask for each channel
    uchar Ga = (static_cast<int>(pow(2, GRBbits[0]))-1) << dG;
    uchar Ra = (static_cast<int>(pow(2, GRBbits[1]))-1) << dR;
    uchar Ba = (static_cast<int>(pow(2, GRBbits[2]))-1) << dB;

    // Get pixels of individual channels in the input image
    uchar B = (*itI)[0];
    uchar G = (*itI)[1];
    uchar R = (*itI)[2];

    // Extract most significant bits for each channel
    uchar C1 = (B & Ba) >> dG;  // blue channel
    uchar C2 = (R & Ra) >> (dG-GRBbits[1]);  // red channel
    uchar C3 = (G & Ga) >> (dG-GRBbits[1]-GRBbits[2]);  // green channel

    // Merge the bit codes
    uchar newPixel = C1 | C2 | C3;

    // Store in the new image
    (*itQ) = (newPixel > 255) ? 255 : newPixel;
  }

  // Reduce number of colors if necessary
  if (num_colors < 256) {
    reduceImageColors(Q, num_colors);
  }
}
