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

#include <queue>
#include <vector>

#include "description/descritores.h"

/*******************************************************************************
Verify a neighbor pixel. To do so, it checks if it is in the correct dimensions
of the image, than compare to the color we are looking for and if was not yet
visited. Then add this pixel to the queue, mark it as visited and increase the
region size.

Requires:
- Mat original image
- int height of the image to check border
- int width of the image to check border
- uchar color we are expect neighbors have
- vector< vector<bool> >* control of which pixels were visited
- queue<Pixel>* queue of pixels to visit
- int64* size of the region we are calculating
*******************************************************************************/
void VerifyNeighborPixel(Mat img, int index_height, int index_width,
                        uchar pixel_color, vector< vector<bool> > *visited,
                        queue<Pixel> *pixels, int *size_region) {
  int height = img.rows, width = img.cols;
  uchar img_color;
  Pixel pix;

  if (index_height >= 0 && index_height < height &&
      index_width  >= 0 && index_width < width) {
    img_color = img.at<uchar>(index_height, index_width);
    if (!(*visited)[index_height][index_width] &&
        img_color == pixel_color) {
      (*visited)[index_height][index_width] = true;
      pix.i = index_height;
      pix.j = index_width;
      pix.color = pixel_color;
      (*size_region)++;
      (*pixels).push(pix);
    }
  }
}

/*******************************************************************************
Finds the neighbors that are not border and have the same color of the pixel on
the begin of the queue (oldest element) and add them into the queue and increase
the size of the current region.

Requires:
- Mat original image
- vector< vector<bool> >* control of which pixels were visited
- queue<Pixel> queue of pixels to visit
- int64* size of the region we are calculating
*******************************************************************************/
void FindNeighbor(Mat img, vector< vector<bool> > *visited,
                  queue<Pixel> *pixels, int *size_region) {
  Pixel pix = (*pixels).front();
  (*pixels).pop();

  int y = pix.i;
  int x = pix.j;
  uchar color = pix.color;
  int up = y-1;
  int down = y+1;
  int left = x-1;
  int right = x+1;

  // Check if the 8-neighbors pixels are not border and have the same color
  VerifyNeighborPixel(img, up, x, color, visited, pixels, size_region);
  VerifyNeighborPixel(img, up, right, color, visited, pixels, size_region);
  VerifyNeighborPixel(img, y, right, color, visited, pixels, size_region);
  VerifyNeighborPixel(img, down, right, color, visited, pixels, size_region);
  VerifyNeighborPixel(img, down, x, color, visited, pixels, size_region);
  VerifyNeighborPixel(img, down, left, color, visited, pixels, size_region);
  VerifyNeighborPixel(img, y, left, color, visited, pixels, size_region);
  VerifyNeighborPixel(img, up, left, color, visited, pixels, size_region);
}

/*******************************************************************************
A color coherence vector (CCV) stores the number of coherent versus incoherent
pixels of each color.

Coherent pixels area a part of some contiguous region while incoherent are not.

Computes two histograms:
- histogram of coherent pixels.
- histogram of incoherent pixels.

1 - Slightly blur the image by replacing pixels values with the average value
    in a 8-neighborhood.
2 - Discretize the colorspace, such that there are only n distinct colors.
3 - Classify pixels as either coherent or incoherent depending on the size of
    its connected component given a threshold.


Requires:
- Mat original image
- Mat* histogram with size equal to 2*colors
  * coherent: [0, colors-1]
  * incoherent: [colors, colors*2-1]
- int number of colors
- int indicate if normalization is necessary (0-None 1-[0,1] 255-[0,255])
- int level of coherency
*******************************************************************************/
void CalculateCCV(Mat img, Mat *features, int number_colors, int normalization,
                  int threshold) {
  int x, y, size_region, height = img.rows, width = img.cols;
  vector< vector<bool> > visited_pixels;
  visited_pixels.resize(height, vector<bool>(width, false));
  Mat ccv_histograms, blur_img;
  queue<Pixel> pixels;
  Pixel pix;

  // 1 - Blur the image
  blur(img, blur_img, Size(3, 3));

  // 2 - Discretize the colorspace to 1/4 of the colors
  // number_colors = static_cast<int>(number_colors/4.0);
  reduceImageColors(&blur_img, number_colors);

  vector<float> coherent(number_colors, 0);
  vector<float> incoherent(number_colors, 0);

  // 3 - For each pixel, classify it as either coherent or incoherent
  for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) {
      // If the current pixel was not visited yet
      if (!visited_pixels[y][x]) {
        pix.i = y;
        pix.j = x;
        pix.color = blur_img.at<uchar>(y, x);
        pixels.push(pix);
        // Mark it as visited
        visited_pixels[y][x] = true;
        // Start a new region for this pixel
        size_region = 1;
        // Find all the neighbors that are not border and have the same color
        while (!pixels.empty()) {
          // If neither neighbor has the same color, call it only one time
          // If a neighbor with the same color is found, call it for the new one
          FindNeighbor(blur_img, &visited_pixels, &pixels, &size_region);
        }
        // If the size of the region is higher than the threshold is coherent
        if (size_region >= threshold) {
          coherent[pix.color] += size_region;
        } else {  // Otherwise, is incoherent
          incoherent[pix.color] += size_region;
        }
      }
    }
  }

  Mat coherent_hist(coherent, true);
  Mat incoherent_hist(incoherent, true);
  // Normalize the vectors and concatenate them on Mat features
  if (normalization != 0) {
    normalize(coherent_hist, coherent_hist, 0, normalization, NORM_MINMAX,
              -1, Mat());
    normalize(incoherent_hist, incoherent_hist, 0, normalization, NORM_MINMAX,
              -1, Mat());
  }
  ccv_histograms.push_back(coherent_hist);
  ccv_histograms.push_back(incoherent_hist);
  ccv_histograms = ccv_histograms.t();  // Transpose to make rowsx1 be 1xcols
  (*features).push_back(ccv_histograms);
  ccv_histograms.release();
}

/*******************************************************************************
Compute color coherent vector for each channel and concatenate them

Requires:
- Mat original image
- Mat* feature vector where to save the histograms
- int number of colors wanted in the image
- int indicate if normalization is necessary (0-None 1-[0,1] 255-[0,255])
- int threshold to indicate the size of the region that is coherent
*******************************************************************************/
void CCV(Mat img, Mat *features, int colors, int normalization, int threshold) {
  int i, img_channels = img.channels();
  vector<Mat> color_ccv(img_channels), channel(img_channels);
  Mat ccv_histograms;

  if (img_channels > 1) reduceImageColors(&img, colors);
  split(img, channel);

  for (i = 0; i < img_channels; i++) {
    CalculateCCV(channel[i], &(color_ccv[i]), colors, normalization, threshold);
    color_ccv[i] = color_ccv[i].t();
    ccv_histograms.push_back(color_ccv[i]);
  }

  ccv_histograms = ccv_histograms.t();
  (*features).push_back(ccv_histograms);
}

/*******************************************************************************
The color histogram is obtained by discretizing the image colors and counting
the number of times each color occurs.

Usually a 64-bins histogram is used.

Requires:
- Mat original image
- Mat* feature vector where to save the histogram
- int number of colors wanted in the image (64)
- int indicate if normalization is necessary (0-None 1-[0,1] 255-[0,255])
*******************************************************************************/
void CalculateGCH(Mat img, Mat *features, int colors, int normalization) {
  int histogram_size[] = {colors};
  float ranges[] = {0, static_cast<float>(colors)};
  const float* histogram_ranges[] = {ranges};
  MatND histogram;
  bool uniform = true, accumulate = false;

  // Calculate the histogram. Be aware that the discretization occur
  // when a bin < 256 is given as input to his function
  calcHist(&img, 1, 0, Mat(), histogram, 1, histogram_size, histogram_ranges,
          uniform, accumulate);

  if (normalization != 0) {
    normalize(histogram, histogram, 0, normalization, NORM_MINMAX, -1, Mat());
  }
  histogram = histogram.t();
  (*features).push_back(histogram);
  histogram.release();
}

/*******************************************************************************
Compute a global color histogram for each channel and concatenate them

Requires:
- Mat original image
- Mat* feature vector where to save the two histograms
- int number of colors wanted in the image
- int indicate if normalization is necessary (0-None 1-[0,1] 255-[0,255])
*******************************************************************************/
void GCH(Mat img, Mat *features, int colors, int normalization) {
  int i, img_channels = img.channels();
  vector<Mat> color_gch(img_channels);
  Mat gch_histograms;
  vector<Mat> channel(img_channels);

  if (img_channels > 1) reduceImageColors(&img, colors);
  split(img, channel);

  for (i = 0; i < img_channels; i++) {
    CalculateGCH(channel[i], &(color_gch[i]), colors, normalization);
    color_gch[i] = color_gch[i].t();
    gch_histograms.push_back(color_gch[i]);
  }

  gch_histograms = gch_histograms.t();
  (*features).push_back(gch_histograms);
}

/*******************************************************************************
Border/Interior pixel classification
1 - Image pixels are classified as border or interior pixels.
    - Border: if it is at the border of the image itself or if at least one of
    its 4-neighbors has a different quantized color.
    - Interior: if its 4-neighbors have the same quantized color.
2 - Compute two color histograms, one for border and other for interior.

Usually 64 bins each color histogram.

Requires:
- Mat original image
- Mat* feature vector where to save the two histograms
    * Borda: [0, colors-1]
    * Interior: [colors, colors*2-1]
- int number of colors wanted in the image
- int indicate if normalization is necessary (0-None 1-[0,1] 255-[0,255])
*******************************************************************************/
void CalculateBIC(Mat img, Mat *features, int colors, int normalization) {
  int height = img.rows, width = img.cols, row, col;
  uchar pixel_color;
  Mat histogram_border = Mat::zeros(1, colors, CV_32FC1);
  Mat histogram_interior = Mat::zeros(1, colors, CV_32FC1);

  // 1- Calculate the histogram while classify pixels as border or interior
  for (row = 0; row < height; row++) {
    for (col = 0; col < width; col++) {
      pixel_color = img.at<uchar>(row, col);
      // If the pixel is not border
      if (row > 0 && col > 0 && col < width-1 && row < height-1) {
        // If all the 4-neighbors has the same color it is interior
        if ((img.at<uchar>(row, col+1) == pixel_color) &&
            (img.at<uchar>(row-1, col) == pixel_color) &&
            (img.at<uchar>(row, col-1) == pixel_color) &&
            (img.at<uchar>(row+1, col) == pixel_color)) {
              histogram_interior.at<float>(0, static_cast<float>(pixel_color))++;
        } else {  // If some neighbor has a different color, is border
          histogram_border.at<float>(0, static_cast<float>(pixel_color))++;
        }
      } else {  // If the current pixel is in the image border
        histogram_border.at<float>(0, static_cast<float>(pixel_color))++;
      }
    }
  }

  //  2- Normalize
  if (normalization != 0) {
    normalize(histogram_border, histogram_border, 0, normalization,
              NORM_MINMAX, -1, Mat());
    normalize(histogram_interior, histogram_interior, 0, normalization,
              NORM_MINMAX, -1, Mat());
  }

  // 3- Concatenate
  histogram_border = histogram_border.t();
  histogram_interior = histogram_interior.t();
  (*features).push_back(histogram_border);
  (*features).push_back(histogram_interior);
}

/*******************************************************************************
Compute a Border/Interior pixel classification for each channel

Requires:
- Mat original image
- Mat* feature vector where to save the two histograms
- int number of colors wanted in the image
- int indicate if normalization is necessary (0-None 1-[0,1] 255-[0,255])
*******************************************************************************/
void BIC(Mat img, Mat *features, int colors, int normalization) {
  int i, img_channels = img.channels();
  vector<Mat> color_bic(img_channels);
  Mat bic_histograms;
  vector<Mat> channel(img_channels);

  if (img_channels > 1) reduceImageColors(&img, colors);
  split(img, channel);

  for (i = 0; i < img_channels; i++) {
    CalculateBIC(channel[i], &(color_bic[i]), colors, normalization);
    // color_bic[i] = color_bic[i].t();
    bic_histograms.push_back(color_bic[i]);
  }

  bic_histograms = bic_histograms.t();
  (*features).push_back(bic_histograms);
}

/*******************************************************************************
Four directions of adjacency as defined for calculation of the Haralick texture
features
*******************************************************************************/
vector<int> NearestNeighborAngle(int row, int col, int distance, int angle) {
  vector<int> neighbor(2, 0);
  neighbor[0] = 0;
  neighbor[1] = 0;

  // Right pixel
  if (angle == 0) {
    neighbor[0] = row;
    neighbor[1] = col + distance;
  }
  // Up-Right pixel
  if (angle == 45) {
    neighbor[0] = row - distance;
    neighbor[1] = col + distance;
  }
  // Up pixel
  if (angle == 90) {
    neighbor[0] = row - distance;
    neighbor[1] = col;
  }
  // Up-Left pixel
  if (angle == 135) {
    neighbor[0] = row - distance;
    neighbor[1] = col - distance;
  }
  return neighbor;
}

/*******************************************************************************
A co-occurrence matriz is a matriz that is defined over an image to be the
distribution of co-occurring values at a given offset (distance).

1 - Count the occurrences and fill the matrix: considers the relation between two
pixels at a time, called the reference and the neighbor pixel.
2 - Make it symmetrical.
3 - Normalize the matrix to turn it into probabilities.

Requires:
- Mat original image
- vector< vector<float> >* co-occurence matrix
- int number of colors wanted in the image
- int distance between neighbors (usually 2)
- int angle (0||45||90||135)
*******************************************************************************/
void CoocurrenceMatrix(Mat img, vector< vector<float> > *co_occurence,
                      int colors, int distance, int angle) {
  int64 number_occurences = 0;
  int color_reference, color_neighbor;
  int row, col, height = img.rows, width = img.cols;
  vector<int> neighbor;

  reduceImageColors(&img, colors);
  (*co_occurence).resize(colors, vector<float>(colors, 0));
  for (row = distance; row < height-distance; ++row) {
    for (col = distance; col < width-distance; ++col) {
      neighbor = NearestNeighborAngle(row, col, distance, angle);
      color_reference = img.at<uchar>(row, col);
      color_neighbor = img.at<uchar>(neighbor[0], neighbor[1]);
      // Symmetry will be achieved if each pixel pair is counted twice
      (*co_occurence)[color_reference][color_neighbor]++;
      number_occurences++;
      (*co_occurence)[color_neighbor][color_reference]++;
      number_occurences++;
    }
  }

  // Normalize to turn it into probabilities, by dividing each entry in the
  // matrix by the sum of pairs
  for (row = 0; row < colors; row++) {
    for (col = 0; col < colors; col++) {
      (*co_occurence)[row][col] /= static_cast<float> (number_occurences);
    }
  }
}

/*******************************************************************************
Image texture refers to local differences in intensity levels. That is why it
needs a GLCM matrix to calculate the statistics.

* Max_probability: stronger response at the co-occurence matrix. Range: [0,1]
* Correlation: describes the correlations between the rows and columns of the
co-occurrence matrix. Range: [-1,1]
* Contrast: measures the local variations in the gray-level co-occurrence
matrix. Range: [0, (colors-1)^2]
* Uniformity: Sum of squared elements. Also known as energy or the angular
second moment. Range: [0,1]
* Homogeneity: measures the closeness of the distribution of elements to the
diagonal. Range: [0,1]
* Entropy: descriptor of randomness. Range:  [0, 2 * log_2 colors]

Requires:
- vector< vector<float> >* GLCM matrix
- Mat* to write the 6 measurements after computing them
*******************************************************************************/
void Haralick6(vector< vector<float> > co_occurence, Mat *features) {
  int i, j;
  double mean_rows = 0, mean_cols = 0, standard_deviation_rows = 0;
  double standard_deviation_cols = 0, entropy = 0, homogeneity = 0;
  double max_probability = 0, correlation = 0, contrast = 0, uniform = 0;
  double variance_rows = 0, variance_cols = 0, p_ij;
  int colors = co_occurence.size();

  vector <double> frequency_rows(colors, 0);
  vector <double> frequency_cols(colors, 0);

  for (i = 0; i < colors; i++) {
    for (j = 0; j < colors; j++) {
      frequency_rows[i] += co_occurence[i][j];
      frequency_cols[i] += co_occurence[j][i];
    }
  }

  for (i = 0; i < colors; i++) {
    mean_rows += static_cast<double>(i) * frequency_rows[i];
    mean_cols += static_cast<double>(i) * frequency_cols[i];
  }

  for (i = 0; i < colors; i++) {
    variance_rows += pow((i - mean_rows), 2.0) * frequency_rows[i];
    variance_cols += pow((i - mean_cols), 2.0) * frequency_cols[i];
  }
  standard_deviation_rows = sqrt(variance_rows);
  standard_deviation_cols = sqrt(variance_cols);

  for (i = 0; i < colors; ++i) {
    for (j = 0; j < colors; ++j) {
      p_ij = co_occurence[i][j];
      // Find the max value in the co-occurence matrix
      if (max_probability < p_ij) {
        max_probability = p_ij;
      }
      // Correlations between the rows and columns of the co-occurrence matrix
      if (standard_deviation_rows != 0 && standard_deviation_cols != 0) {
        correlation += (double) p_ij * (((i-mean_rows)*(j-mean_cols)) / (double)
                      (standard_deviation_rows*standard_deviation_cols));
      }
      // Local variations in the gray-level co-occurrence matrix
      contrast += pow(i - j, 2.0) * p_ij;
      // Sum of squared elements
      uniform += pow(p_ij, 2.0);
      // Closeness of the distribution of elements to the diagonal
      homogeneity += (double) p_ij / (double) (1.0 + pow(i-j, 2.0));
      // Randomness
      if (p_ij != 0) {
        entropy += (double) p_ij * (double) log2(p_ij);
      }
    }
  }

  (*features).create(1, 6, CV_32FC1);
  (*features) = Scalar::all(0.0);

  entropy = -entropy;
  (*features).at<float>(0, 0) = (float) max_probability;
  (*features).at<float>(0, 1) = (float) correlation;
  (*features).at<float>(0, 2) = (float) contrast;
  (*features).at<float>(0, 3) = (float) uniform;
  (*features).at<float>(0, 4) = (float) homogeneity;
  (*features).at<float>(0, 5) = (float) entropy;
}

/*******************************************************************************
Texture descriptor: Haralick feature extraction

1 - Calculates a GLCM rotationally invariant
2 - Extract 6 Haralick features from it

Requires:
- Mat original image
- Mat* feature vector where to write the features
- int number of colors wanted in the image
- int indicate if normalization is necessary (0-None 1-[0,1] 255-[0,255])
*******************************************************************************/
void CalculateHARALICK(Mat img, Mat *features, int colors, int normalization) {
  vector< vector<float> > GLCM_0, GLCM_45, GLCM_90, GLCM_135, GLCM;
  int distance, i, j;

  distance = 1;
  CoocurrenceMatrix(img, &GLCM_0, colors, distance, 0);
  CoocurrenceMatrix(img, &GLCM_45, colors, distance, 45);
  CoocurrenceMatrix(img, &GLCM_90, colors, distance, 90);
  CoocurrenceMatrix(img, &GLCM_135, colors, distance, 135);

  // The GLCM matrix is the average of four matrixes with different directions
  GLCM.resize(colors, vector<float>(colors, 0));
  for (i = 0; i < colors; ++i) {
    for (j = 0; j < colors; ++j) {
      GLCM[i][j] =
        (GLCM_0[i][j] + GLCM_45[i][j] + GLCM_90[i][j] + GLCM_135[i][j]) / 4.0;
    }
  }

  Haralick6(GLCM, features);
}

/*******************************************************************************
Texture descriptor: Haralick feature extraction

Requires:
- Mat original image
- Mat* feature vector where to write the features
- int number of colors wanted in the image
- int indicate if normalization is necessary (0-None 1-[0,1] 255-[0,255])
*******************************************************************************/
void HARALICK(Mat img, Mat *features, int colors, int normalization) {
  int i, img_channels = img.channels();
  vector<Mat> color_haralick(img_channels);
  Mat haralick_histograms;
  vector<Mat> channel(img_channels);

  if (img_channels > 1) reduceImageColors(&img, colors);
  split(img, channel);

  for (i = 0; i < img_channels; i++) {
    CalculateHARALICK(channel[i], &(color_haralick[i]), colors, normalization);
    color_haralick[i] = color_haralick[i].t();
    haralick_histograms.push_back(color_haralick[i]);
  }

  haralick_histograms = haralick_histograms.t();
  (*features).push_back(haralick_histograms);
}

/*******************************************************************************
Used in Auto-correlogram of colors Descritor

Returns the neighbors in a 8-neighborhood.

Requires:
- int y position [0..height/rows]
- int x position [0..width/cols]
- int number of colors wanted in the image
*******************************************************************************/
vector < vector<int> > ChessboardNeighbors(int row, int col, int distance) {
  vector < vector<int> > neighbors;
  int up = row - distance, down = row + distance;
  int left = col - distance, right = col + distance;

  neighbors.push_back({up, col});
  // neighbors.push_back({up, right});
  neighbors.push_back({row, right});
  // neighbors.push_back({down, right});
  neighbors.push_back({down, col});
  // neighbors.push_back({down, left});
  neighbors.push_back({row, left});
  // neighbors.push_back({up, left});

  return neighbors;
}

/*******************************************************************************
Auto-correlogram of colors Descritor

Capture the spacial correlation between identical colors given a set of distance
values. The features vector consist in concatenation of auto-correlograms, one
for each distance.

Requires:
- Mat original image
- Mat* feature vector where to write the colors*distances_number features
- int number of colors wanted in the image
- int indicate if normalization is necessary (0-None 1-[0,1] 255-[0,255])
- vector<int> set of distances
*******************************************************************************/
void CalculateACC(Mat I, Mat *features, int colors, int normalization,
        vector<int> distances) {
  int i, j, d, current_distance, chess;
  vector < vector<int>> neighbors;
  uchar current_pixel, neighbor_color;

  reduceImageColors(&I, colors);
  Mat acc_correlogram, autocorrelogram(colors, 1, CV_32FC1);
  // For each given distance in 'distances' set
  for (d = 0; d < static_cast<int>(distances.size()); ++d) {
    autocorrelogram = Scalar::all(0.0);
    current_distance = distances[d];
    // For each pixel
    for (i = current_distance; i < I.rows - current_distance; ++i) {
      for (j = current_distance; j < I.cols - current_distance; ++j) {
        current_pixel = I.at<uchar>(i, j);
        // Find the 8-neighbors in a distance
        neighbors = ChessboardNeighbors(i, j, current_distance);
        // For each neighbor
        for (chess = 0; chess < static_cast<int>(neighbors.size()); ++chess) {
          neighbor_color =
            I.at<uchar>(neighbors[chess][0], neighbors[chess][1]);
          // If both pixels have the same color, plus one in the correlogram
          if (current_pixel == neighbor_color) {
            autocorrelogram.at<float>(static_cast<int>(current_pixel), 0)++;
          }
        }
      }
    }
    // Normalize for each distance, not when already concatenated
    if (normalization != 0) {
      normalize(autocorrelogram, autocorrelogram, 0, normalization, NORM_MINMAX,
                -1, Mat());
    }
    acc_correlogram.push_back(autocorrelogram);
  }
  acc_correlogram = acc_correlogram.t();
  (*features).push_back(acc_correlogram);
  acc_correlogram.release();
  autocorrelogram.release();
}

/*******************************************************************************
Auto-correlogram of colors descriptor

Requires:
- Mat original image
- Mat* feature vector where to write the colors*distances_number features
- int number of colors wanted in the image
- int indicate if normalization is necessary (0-None 1-[0,1] 255-[0,255])
- vector<int> set of distances
*******************************************************************************/
void ACC(Mat img, Mat *features, int colors, int normalization,
        vector<int> distances) {
  int i, img_channels = img.channels();
  vector<Mat> color_acc(img_channels), channel(img_channels);
  Mat acc_histograms;

  if (img_channels > 1) reduceImageColors(&img, colors);
  split(img, channel);

  for (i = 0; i < img_channels; i++) {
    CalculateACC(channel[i], &(color_acc[i]), colors, normalization,
      distances);
    color_acc[i] = color_acc[i].t();
    acc_histograms.push_back(color_acc[i]);
  }

  acc_histograms = acc_histograms.t();
  (*features).push_back(acc_histograms);
}

/*******************************************************************************
Create a lookup table to LBP function checks if the pattern is uniform

Some binary patterns occur more commonly than others
A binary code is binary if contains at most two 0-1 or 1-0 transitions
*******************************************************************************/
vector<int> initUniform() {
  int index = 0, i = 0, b = 0, count = 0, c = 0;
  vector<int> lookup(256);

  for (i = 0; i < 256; i++) {
    b = (i >> 1) | (i << 7 & 0xff);
    c = i ^ b;
    // Count the number of transitions
    for (count = 0; c; count++) {
      c &= c-1;  // clears the LSB
    }
    // Each uniform code is assigned to an index
    if (count <= 2) {
      lookup[i] = index;
      index++;
    } else {  // All non uniform codes are assigned to a single bin
      lookup[i] = 58;
    }
  }
  return lookup;
}

/*******************************************************************************
LBP Descriptor using uniform patterns

It is a histogram of quantized LBPs pooled in a local image neighborhood.
This version is an extension of the original LBP by using the proposed..

1- Divide the examined window into cells
2- For each pixel in a cell, compare the pixel to its neighbors. This gives a
  8-digit binary code
3- Compute the histogram over the cell, of the frequency of each number occuring
4- Normalize the histogram
5- Concatenated the histograms of all cells

Requires
- Mat original image
- Mat features vector in which perform the operations
- int number of colors
- int indicate if normalization is necessary (0-None 1-[0,1] 255-[0,255])
*******************************************************************************/
void CalculateLBP(Mat img, Mat *features, int colors, int normalization) {
  int bin, cellWidth, cellHeight, i, j, bitString, grid, bias, row, col;
  int histogram_size[] = {59};
  float center, ranges[] = {0, 59};
  const float* histogram_ranges[] = {ranges};
  bool uniform = true, accumulate = false;
  Size cell;
  MatND histogram, lbp_histograms;
  Mat codes = Mat::zeros(img.rows, img.cols, CV_32FC1);
  vector<int> lookup = initUniform();

  // For each pixel in a cell, compare it to each of its 8 neighbors
  for (row = 1; row < img.rows - 1; row++) {
    for (col = 1; col < img.cols - 1; col++) {
      // Where the center pixel's value is greater than the neighbor's, write 1
      // Otherwise, it remains 0.
      bitString = 0;
      center = img.at<uchar>(row, col);
      // Start from the one to the right in anti-clockwise order
      if (img.at<uchar>(row, col+1)   >= center) bitString |= 0x1 << 0;
      if (img.at<uchar>(row-1, col+1) >= center) bitString |= 0x1 << 1;
      if (img.at<uchar>(row-1, col)   >= center) bitString |= 0x1 << 2;
      if (img.at<uchar>(row-1, col-1) >= center) bitString |= 0x1 << 3;
      if (img.at<uchar>(row, col-1)   >= center) bitString |= 0x1 << 4;
      if (img.at<uchar>(row+1, col-1) >= center) bitString |= 0x1 << 5;
      if (img.at<uchar>(row+1, col)   >= center) bitString |= 0x1 << 6;
      if (img.at<uchar>(row+1, col+1) >= center) bitString |= 0x1 << 7;
      // This gives a 8-digit binary number corresponding to the binary code
      // Instead of using the resulting bitString [0-256], we use a lookup table
      bin = lookup[bitString];
      codes.at<float>(row-1, col-1) = bin;
    }
  }

  // namedWindow("LBP Image", WINDOW_AUTOSIZE);
  // imshow("LBP Image", (codes/255.0)*4);
  // waitKey(0);

  Mat lbp = Mat::zeros(img.rows + 2, img.cols + 2, CV_8U);
  copyMakeBorder(codes, lbp, 1, 1, 1, 1, BORDER_REPLICATE);

  grid = 2;
  cellWidth = lbp.cols/grid;
  cellHeight = lbp.rows/grid;

  bias = 0;
  if (cellWidth * grid < lbp.cols - 1) {
    bias = 1;
  }

  for (i = 0; i < grid; i++) {
    for (j = 0; j < grid; j++) {
      Mat cell =
        lbp(Rect(i*cellWidth+bias, j*cellHeight+bias, cellWidth, cellHeight));
      // Calculate the histogram for this cell
      calcHist(&cell, 1, 0, Mat(), histogram, 1, histogram_size,
        histogram_ranges, uniform, accumulate);

      if (normalization != 0) {
        normalize(histogram, histogram, 0, normalization, NORM_MINMAX, -1,
          Mat());
      }
      lbp_histograms.push_back(histogram);
      histogram.release();
    }
  }
  // PlotHistogram(lbp_histograms);
  lbp_histograms = lbp_histograms.t();  // Transpose to make rowsx1 be 1xcols
  (*features).push_back(lbp_histograms);
  lbp_histograms.release();
}

/*******************************************************************************
LBP Descriptor using uniform patterns

Requires
- Mat original image
- Mat features vector in which perform the operations
- int number of colors
- int indicate if normalization is necessary (0-None 1-[0,1] 255-[0,255])
*******************************************************************************/
void LBP(Mat img, Mat *features, int colors, int normalization) {
  int i, img_channels = img.channels();
  vector<Mat> color_lbp(img_channels);
  Mat lbp_histograms;
  vector<Mat> channel(img_channels);

  if (img_channels > 1) reduceImageColors(&img, colors);
  split(img, channel);

  for (i = 0; i < img_channels; i++) {
    CalculateLBP(channel[i], &(color_lbp[i]), colors, normalization);
    color_lbp[i] = color_lbp[i].t();
    lbp_histograms.push_back(color_lbp[i]);
  }

  lbp_histograms = lbp_histograms.t();
  (*features).push_back(lbp_histograms);
}

/*******************************************************************************
Orientation Descriptor - Histogram of Oriented Gradients

numFeatures = hog.nbins * (blockSize.width/cellSize.width)
  (blockSize.height/cellSize.height)
  ((winSize.width - blockSize.width)/blockStride.width + 1)
  ((winSize.height - blockSize.height)/ blockStride.height + 1);

Requires
- Mat original image
- Mat features vector in which perform the operations
- int number of colors
- int indicate if normalization is necessary (0-None 1-[0,1] 255-[0,255])
*******************************************************************************/
void CalculateHOG(Mat img, Mat *features, int normalization) {
  HOGDescriptor hog;
  vector<float> hogFeatures;
  vector<Point> locs;
  int i, cellSize;
  Mat new_image;
  Size new_size = Size(256, 256);

  img.copyTo(new_image);
  cv::resize(new_image, new_image, new_size);

  hog.winSize = new_size;
  cellSize = 16;
  hog.blockSize = Size(cellSize*2, cellSize*2);
  hog.blockStride = Size(cellSize, cellSize);
  hog.cellSize = Size(cellSize, cellSize);

  hog.compute(new_image, hogFeatures);

  (*features).create(1, hogFeatures.size(), CV_32FC1);
  (*features) = Scalar::all(0.0);
  for (i = 0; i < static_cast<int>(hogFeatures.size()); i++) {
      (*features).at<float>(0, i) = (float) hogFeatures.at(i);
  }
}

/*******************************************************************************
Orientation Descriptor - Histogram of Oriented Gradients

Requires
- Mat original image
- Mat features vector in which perform the operations
- int number of colors
- int indicate if normalization is necessary (0-None 1-[0,1] 255-[0,255])
*******************************************************************************/
void HOG(Mat img, Mat *features, int colors, int normalization) {
  int i, img_channels = img.channels();
  vector<Mat> color_hog(img_channels);
  Mat hog_histograms;
  vector<Mat> channel(img_channels);

  if (img_channels > 1) reduceImageColors(&img, colors);
  split(img, channel);

  for (i = 0; i < img_channels; i++) {
    CalculateHOG(channel[i], &(color_hog[i]), normalization);
    color_hog[i] = color_hog[i].t();
    hog_histograms.push_back(color_hog[i]);
  }

  hog_histograms = hog_histograms.t();
  (*features).push_back(hog_histograms);
}

/*******************************************************************************
Shape Descriptors - Contour Extraction

Calculate the biggest contour and extract the mass centers, number of pixels
inside the contour, perimeter and approximated area.

Requires
- Mat original image
- Mat features vector in which perform the operations
- int number of colors
- int indicate if normalization is necessary (0-None 1-[0,1] 255-[0,255])
*******************************************************************************/
void CalculateContour(Mat img, Mat *features, int normalization) {
  vector<vector<Point> > contours;
  vector<Point> approx;
  vector<Vec4i> hierarchy;
  Mat bin;
  int i, biggestAreaIndex = 0;
  double area = 0.0, areaApprox, perimeter, biggestArea = 0.0;
  Moments mu;
  Point2f mc;

  threshold(img, bin, 100, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
  findContours(bin, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

  for (i = 0; i < static_cast<int>(contours.size()); i++) {
    area = contourArea(contours[i], false);
    if (area > biggestArea) {
      biggestArea = area;
      biggestAreaIndex = i;
    }
  }

  // Mat contourImage(bin.size(), CV_8UC3, Scalar(0, 0, 0));
  // for (i = 0; i < static_cast<int>(contours.size()); i++) {
  //   Scalar color(rand()&255, rand()&255, rand()&255);
  //   drawContours(contourImage, contours, i, color);
  // }
  // namedWindow("All", 1);
  // imshow("All", contourImage);
  // waitKey(0);

  // Mat biggestAreaImg(bin.size(), CV_8UC3, Scalar(0, 0, 0));
  // Scalar color(255, 0, 0);
  // drawContours(biggestAreaImg, contours, biggestAreaIndex, color, 1, 8,
  //   hierarchy);
  // namedWindow("Biggest Contour", 1);
  // imshow("Biggest Contour", biggestAreaImg);
  // waitKey(0);

  (*features).create(1, 5, CV_32FC1);
  (*features) = Scalar::all(0.0);

  // Get the moments
  mu = moments(contours[biggestAreaIndex], false);
  //  Get the mass centers:
  mc = Point2f(mu.m10/mu.m00, mu.m01/mu.m00);
  (*features).at<float>(0, 0) = mc.x;
  (*features).at<float>(0, 1) = mc.y;

  // Number of pixels inside the contour
  (*features).at<float>(0, 2) = biggestArea;

  // Contour perimeter
  perimeter = arcLength(contours[biggestAreaIndex], true);
  (*features).at<float>(0, 3) = perimeter;

  // Remove small curves by approximating the contour more to a straight line
  approxPolyDP(contours[biggestAreaIndex], approx, 0.1*perimeter, true);
  areaApprox = contourArea(approx);
  (*features).at<float>(0, 4) = areaApprox;
}

/*******************************************************************************
Shape Descriptors - Contour Extraction

Requires
- Mat original image
- Mat features vector in which perform the operations
- int number of colors
- int indicate if normalization is necessary (0-None 1-[0,1] 255-[0,255])
*******************************************************************************/
void contourExtraction(Mat img, Mat *features, int colors, int normalization) {
  int i, img_channels = img.channels();
  vector<Mat> color_contour(img_channels);
  Mat contour_histograms;
  vector<Mat> channel(img_channels);

  if (img_channels > 1) reduceImageColors(&img, colors);
  split(img, channel);

  for (i = 0; i < img_channels; i++) {
    CalculateContour(channel[i], &(color_contour[i]), normalization);
    color_contour[i] = color_contour[i].t();
    contour_histograms.push_back(color_contour[i]);
  }

  contour_histograms = contour_histograms.t();
  (*features).push_back(contour_histograms);
}

// /****************************************************************************
//  SURF - extract Speeded Up Robust Features
//
//      Input
//          Mat original image
//          Mat features vector in which perform the operations
// ***************************************************************************/
// void surf(Mat img, Mat *features){
//
//     cv::initModule_nonfree();
//     vector<KeyPoint> keypoints;
//     int i, minHessian = 500;
//
//     SurfFeatureDetector detector(minHessian);
//     SurfDescriptorExtractor extractor;
//
//     detector.detect(img, keypoints);
//
//     Mat imgKey;
//     drawKeypoints(img, keypoints, imgKey, Scalar::all(-1),
//      DrawMatchesFlags::DEFAULT);
//     // imshow("Keypoints", imgKey);
//     // waitKey(0);
//     Mat descriptors;
//     extractor.compute(img, keypoints, descriptors);
//
//     cout << descriptors.size();
//     (*features).create(descriptors.rows, 1, CV_32FC1);
//     (*features) = Scalar::all(0);
//
//     for(i = 0; i < descriptors.rows; i++){
//         (*features).at<float>(0,i) = descriptors.at<float>(i,0);
//     }
// }

/*******************************************************************************
Fisher Vectors Encoding

Input
Mat original image
Mat features vector in which perform the operations

- Extract features with SIFT
- Calculate a Gaussian Mixture Model (GMM)
- Obtain the Fisher Vector encoding of a set of features
*******************************************************************************/
// void fisherVectors(Mat img, Mat *feature){
//   float *means, *covariances, *priors, *posteriors, *enc;
//   int numClusters = 30;//, numData = img.cols,
//   int dimension = 3;
//
//   vector<KeyPoint> keypoints;
//   Mat descriptors;
//   float* data;
//   vl_size numData, dimension, numClusters;
//   // create a GMM object and cluster input data to get means, covariances
//   // and priors of the estimated mixture
//   gmm = vl_gmm_new (VL_TYPE_FLOAT) ;
//   VLGMM** gmm;
//   vl_gmmean_colsluster (gmm, data, dimension, numData, numClusters);
//
//   // allocate space for the encoding
//   enc = vl_malloc(sizeof(float) * 2 * dimension * numClusters);
//   // run fisher encoding
//   vl_fisher_encode(
//       enc, VL_F_TYPE,
//       vl_gmm_get_means(gmm), dimension, numClusters,
//       vl_gmm_get_covariances(gmm),
//       vl_gmm_get_priors(gmm),
//       dataToEncode, numDataToEncode,
//       VL_FISHER_FLAG_IMPROVED
//   );
//
//       (*features).create(enc.size(), 1, CV_32FC1);
//       (*features) = Scalar::all(0);
//
//       for(i = 0; i < enc.size(); i++){
//           (*features).at<float>(0,i) = enc.at(i);
//       }
// }
