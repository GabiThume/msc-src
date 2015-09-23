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

#include "utils/funcoesAux.h"

/*******************************************************************************
    Plot in a window the input histogram

    Requires:
    - Mat histogram :
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
    - Mat image to be processed
    - double gamma correction parameter
*******************************************************************************/
void correctGamma(Mat *I, double gamma) {

    Mat result, lut_matrix(1, 256, CV_8UC1);
    uchar *ptr = lut_matrix.ptr();
    double iGamma = 1.0 / gamma;
    int i, img_channels = (*I).channels();

    // Create a lookup table with a grayscale color ajusted for each color
    for (i = 0; i < 256; i++ ) {
        ptr[i] = (uchar)(pow((double)i/255.0, iGamma) * 255.0);
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
void reduceImageColors(Mat *img, int nColors) {

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


/*******************************************************************************
    Remove null columns in feature space (Under construction)

    Requires:
    - Mat features
*******************************************************************************/
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
