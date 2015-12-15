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
#include <bitset>
#include "quantization/quantization.h"
/*******************************************************************************
Image quantization by Gamma corrected Intensity

GrayscaleImage = (R + G + B)*1/3

Requires:
- I: image to be converted into grayscale
- Q: image to store quantized version
- num_colors: number of colors after quantization
*******************************************************************************/
void GrayscaleConversion::Intensity(Mat I, Mat *Q, int num_colors) {
  if (I.channels() == 1) return;

  vector<Mat> imColors(3);
  (*Q).create(I.size(), CV_8UC1);

  // Split input image into separated channels
  split(I, imColors);

  // Compute grayscale image
  (*Q) = imColors[0]/3.0 + imColors[1]/3.0 + imColors[2]/3.0;

  // Gamma correction after quantization
  GrayscaleConversion::correctGamma(Q, 2.2);

  // Reduce number of colors if necessary
  if (num_colors < 256) {
    GrayscaleConversion::reduceImageColors(Q, num_colors);
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
void GrayscaleConversion::Gleam(Mat I, Mat *Q, int num_colors) {
  if (I.channels() == 1) return;

  vector<Mat> imColors(3);
  (*Q).create(I.size(), CV_8UC1);

  split(I, imColors);

  GrayscaleConversion::correctGamma(&(imColors[0]), 2.2);
  GrayscaleConversion::correctGamma(&(imColors[1]), 2.2);
  GrayscaleConversion::correctGamma(&(imColors[2]), 2.2);

  // Sum 1/3 of each channel using gamma corrected pixels
  (*Q) = imColors[0]/3.0 + imColors[1]/3.0 + imColors[2]/3.0;

  // Reduce number of colors if necessary
  if (num_colors < 256) {
    GrayscaleConversion::reduceImageColors(Q, num_colors);
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
void GrayscaleConversion::Luminance(Mat I, Mat *Q, int num_colors) {
  if (I.channels() == 1) return;

  vector<Mat> imColors(3);
  (*Q).create(I.size(), CV_8UC1);

  // split image in RGB channels
  split(I, imColors);

  // Weighted combination considering input image is BGR
  (*Q) = 0.299*imColors[2] + 0.587*imColors[1] + 0.114*imColors[0];

  // Gamma correction
  GrayscaleConversion::correctGamma(Q, 2.2);

  // Reduce number of colors if necessary
  if (num_colors < 256) {
    GrayscaleConversion::reduceImageColors(Q, num_colors);
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
void GrayscaleConversion::Luma(Mat I, Mat *Q, int num_colors) {
  if (I.channels() == 1) return;

  vector<Mat> imColors(3);
  (*Q).create(I.size(), CV_8UC1);

  // split image in RGB channels
  split(I, imColors);

  GrayscaleConversion::correctGamma(&(imColors[0]), 2.2);
  GrayscaleConversion::correctGamma(&(imColors[1]), 2.2);
  GrayscaleConversion::correctGamma(&(imColors[2]), 2.2);

  // Weighted combination considering input image is BGR
  (*Q) = 0.2126*imColors[2] + 0.7152*imColors[1] + 0.0722*imColors[0];

  // Reduce number of colors if necessary
  if (num_colors < 256) {
    GrayscaleConversion::reduceImageColors(Q, num_colors);
  }
}

/*******************************************************************************
Image quantization using the Most Significant Bits (MSB)

Calculate the most significant bits of color channels, in order to
emphasize the chromatic differences. Based on the range of stimulus of
the rod cells for visible light wavelenghts. The order of preference is
G, R and B: GGGRRRBB for 256 colors and GGRRBB00 for 64 colors.

Requires
- I: image to be converted
- Q: image to store quantized version
- num_colors: number of colors after quantization
*******************************************************************************/
void GrayscaleConversion::MSB(Mat I, Mat *Q, int num_colors) {
  if (I.channels() == 1) return;

  int bitsc, rest, k;
  MatIterator_<Vec3b> itI, endI;
  MatIterator_<uchar> itQ, endQ;
  uchar dG, dR, dB, green_mask, red_mask, blue_mask, B, G, R, green_msb;
  uchar red_msb, blue_msb;
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

  dG = (8-GRBbits[0]);
  dR = (8-GRBbits[1]);
  dB = (8-GRBbits[2]);
  // obtain mask for each channel
  green_mask = (static_cast<int>(pow(2, GRBbits[0]))-1) << dG;
  red_mask   = (static_cast<int>(pow(2, GRBbits[1]))-1) << (dR - GRBbits[0]);
  blue_mask  = (static_cast<int>(pow(2, GRBbits[2]))-1)
    << (dB - (GRBbits[0]+GRBbits[1]));

  itQ = (*Q).begin<uchar>();
  endQ = (*Q).end<uchar>();
  itI = I.begin<Vec3b>();
  endI = I.end<Vec3b>();
  for (; itI != endI; ++itI, ++itQ) {
    // Get pixels of individual channels in the input image
    B = (uchar) (*itI)[0];
    G = (uchar) (*itI)[1];
    R = (uchar) (*itI)[2];

    // Extract most significant bits for each channel
    // The order of preference is G, R and then B.
    green_msb = (G & green_mask);
    red_msb   = (R & red_mask);
    blue_msb  = (B & blue_mask);

    // Merge the bit codes and store in the new image
    (*itQ) = saturate_cast<uchar>(green_msb | red_msb | blue_msb);

    // std::bitset<8> b(B), g(G), r(R);
    // cout << "Color: G " << g << " R " << r << " B " << b << endl;
    // std::bitset<8> gm(green_msb), rm(red_msb), bm(blue_msb);
    // cout << "Masks: G " << gm << " R " << rm << " B " << bm << endl;
    // std::bitset<8> p(new_color);
    // cout << "New pixel: " << p << " : " << (int) new_color << endl;
  }
  if (num_colors < 256) {
    GrayscaleConversion::reduceImageColors(Q, num_colors);
  }
}

/*******************************************************************************
Image quantization using the Most Significant Bits (MSB) modified with Tiago
Santana de Nazaré's idea.

Calculate the most significant bits of color channels, in order to
emphasize the chromatic differences. Based on the range of stimulus of
the rod cells for visible light wavelenghts. The order of preference is
G, R and B: GRBGRBGR for 256 colors and GRBGRB00 for 64 colors. -> Modification

Requires
- I: image to be converted
- Q: image to store quantized version
- num_colors: number of colors after quantization
*******************************************************************************/
void GrayscaleConversion::MSBModified(Mat I, Mat *Q, int num_colors) {
  if (I.channels() == 1) return;

  int bitsc, rest, k, color, b;
  MatIterator_<Vec3b> itI, endI;
  MatIterator_<uchar> itQ, endQ;
  uchar green_mask, red_mask, blue_mask, B, G, R, green_msb, red_msb, blue_msb;
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

  // Calculates the mask in a different way: GRBGRBGR
  uchar masks[3] = {0, 0, 0};
  for (b = 7, color = 0; b >= (8-bitsc); color++) {
    if (color == 3) color = 0;
    if (GRBbits[color] > 0) {
      masks[color] = masks[color] | (1 << b);
      b--;
      GRBbits[color]--;
    }
  }
  green_mask = masks[0], red_mask = masks[1], blue_mask = masks[2];

  itQ = (*Q).begin<uchar>();
  endQ = (*Q).end<uchar>();
  itI = I.begin<Vec3b>();
  endI = I.end<Vec3b>();
  for (; itI != endI; ++itI, ++itQ) {
    // Get pixels of individual channels in the input image
    B = (uchar) (*itI)[0];
    G = (uchar) (*itI)[1];
    R = (uchar) (*itI)[2];
    // Extract most significant bits for each channel
    // The order of preference is G, R and then B.
    green_msb = (G & green_mask);
    red_msb   = (R & red_mask);
    blue_msb  = (B & blue_mask);
    // Merge the bit codes and store in the new image
    (*itQ) = saturate_cast<uchar>(green_msb | red_msb | blue_msb);
  }
  if (num_colors < 256) {
    GrayscaleConversion::reduceImageColors(Q, num_colors);
  }
}



/*******************************************************************************
    Plot in a window the input histogram

    Requires:
    - Mat histogram :
*******************************************************************************/
void GrayscaleConversion::PlotHistogram(Mat hist) {
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
    - Mat image to be processed
    - double gamma correction parameter
*******************************************************************************/
void GrayscaleConversion::correctGamma(Mat *I, double gamma) {

    Mat result, lut_matrix(1, 256, CV_8UC1);
    uchar *ptr = lut_matrix.ptr();
    double iGamma = 1.0 / gamma;
    int i, img_channels = (*I).channels();

    // Create a lookup table with a grayscale color ajusted for each color
    for (i = 0; i < 256; i++ ) {
        ptr[i] = saturate_cast<uchar>(pow((double)i/255.0, iGamma) * 255.0);
    }

    vector<Mat> channel(img_channels);
    split((*I), channel);

    for (i = 0; i < img_channels; i++) {
      // Apply the correction in lutMatrix on input img
      LUT(channel[i], lut_matrix, channel[i]);
    }

    merge(channel, (*I));
  }

/*******************************************************************************
    Reduce the number of colors in a single channel image

    Requires:
    - I: input image to be modified
    - nColors: final number of colors
*******************************************************************************/
void GrayscaleConversion::reduceImageColors(Mat *img, int nColors) {

	double min = 0, max = 0, stretch;
	Point maxLoc, minLoc;
  int i, img_channels = (*img).channels();
  vector<Mat> channel(img_channels);
  split((*img), channel);

  nColors = (nColors > 256) ? 256 : nColors;

  for (i = 0; i < img_channels; i++) {
    minMaxLoc(channel[i], &min, &max, &minLoc, &maxLoc);
    stretch = ((double)((nColors -1)) / (max - min));
    channel[i] = channel[i] - min;
    channel[i] = channel[i] * stretch;
  }

  merge(channel, (*img));
}

void GrayscaleConversion::convert(int method, Mat img, Mat *gray) {

  switch (method) {
    case 1:
      GrayscaleConversion::Intensity(img, gray, colors);
      break;
    case 2:
      GrayscaleConversion::Luminance(img, gray, colors);
      break;
    case 3:
      GrayscaleConversion::Gleam(img, gray, colors);
      break;
    case 4:
      GrayscaleConversion::MSB(img, gray, colors);
      break;
    case 5:
      GrayscaleConversion::MSBModified(img, gray, colors);
      break;
    case 6: // Keep in BRG
      break;
    case 7: // Convert BGR -> HSV
      cv::cvtColor(img, (*gray), CV_BGR2HSV);
      break;
    default:
      cout << "Error: quantization method " << method;
      cout << " does not exists." << endl;
  }
  // namedWindow("Display window", WINDOW_AUTOSIZE );
  // imshow("Grayscale Image", *gray);
  // waitKey(0);
}
