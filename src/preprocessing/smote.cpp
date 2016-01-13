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

Based on paper's "SMOTE: Synthetic Minority Over-sampling Technique" pseudo-code
*/

#include <vector>
#include "preprocessing/smote.h"

/* Generate the synthetic samples to over-sample the minority class */
void SMOTE::populate(cv::Mat minority, cv::Mat neighbors, cv::Mat *synthetic, int *index,
    int amountSmote, int i, int nearestNeighbors) {
  int attributes, nn, attr;
  double gap, dif, attrOriginal, neighbor;
  std::vector<int> vectorRand;
  cv::Size s = minority.size();
  attributes = s.width;

  /* amountSmote is how many synthetic samples need to be generated for i */
  while (amountSmote != 0) {
    /* Choose randomly one of the nearest neighbors of i */
    /* The first is the sample itself */
    nn = 1 + (rand() % (nearestNeighbors-1));
    if (!count(vectorRand.begin(), vectorRand.end(), nn)) {
      vectorRand.push_back(nn);
      neighbor = neighbors.at<float>(i, nn);

      for (attr = 0; attr < attributes; attr++) {
        attrOriginal = minority.at<float>(i, attr);
        /* The difference between the feature std::vector under and its
        nearest neighbor*/
        dif = minority.at<float>(neighbor, attr) - attrOriginal;
        /* Multiply this difference with a number between 0 and 1 */
        gap = rand()/static_cast<double>(RAND_MAX);
        /* Add it to the original feature std::vector */
        (*synthetic).at<float>((*index), attr) = attrOriginal + gap*dif;
      }
      (*index)++;
      amountSmote = amountSmote - 1;
    }
  }
}

/* Compute the nearest neighbors */
void SMOTE::computeNeighbors(cv::Mat minority, int nearestNeighbors,
    cv::Mat *neighbors) {
  cv::Size s = minority.size();
  int i, k = nearestNeighbors, max, index = 0;
  cv::Mat classes(s.height, 1, CV_32FC1);

  for (i = 0; i < s.height; i++) {
    classes.at<float>(i, 0) = index;
    index++;
  }

  cv::KNearest knn(minority, classes);
  max = (minority.rows > knn.get_max_k() ? knn.get_max_k() : minority.rows);
  knn.find_nearest(minority, (k > max ? max : k), 0, 0, neighbors, 0);

  // for (i = 0; i < (*neighbors).size().height; i++){
  //   std::cout << std::endl <<  nearestNeighbors << " vizinhos de i: " << i << std::endl;
  //   for (int x = 0; x < (*neighbors).size().width; x++){
  //     std::cout << (*neighbors).at<float>(i, x) << " ";
  //   }
  // }

  classes.release();
}

/* Synthetic Minority Over-sampling Technique */
cv::Mat SMOTE::smote(cv::Mat minority, int amountSmote, int nearestNeighbors) {
  cv::Size s = minority.size();
  int samples, pos, index = 0;
  int minoritySamples = s.height;
  int attributes = s.width;
  std::vector<int> vectorRand;
  cv::Mat newMinority;
  if (amountSmote == 0) return cv::Mat();

  std::cout << "\n---------------------------------------------------------" << std::endl;
  std::cout << "SMOTE generation of samples to rebalance classes" << std::endl;
  std::cout << "---------------------------------------------------------" << std::endl;

  // The first neighbor is the sample itself
  nearestNeighbors =  nearestNeighbors + 1 < minority.rows ? nearestNeighbors + 1 : minority.rows;

  minoritySamples = amountSmote;
  newMinority.create(minoritySamples, attributes, CV_32FC1);
  samples = 0;
  /* If amount to smote is less than 100%, randomize the minority class
  samples as only a random percent of them will be SMOTEd */
  if (amountSmote < s.height) {
    while (samples < minoritySamples) {
      /* Generate a random position for the minority class samples */
      pos = rand() % (s.height);
      if (!count(vectorRand.begin(), vectorRand.end(), pos)) {
        vectorRand.push_back(pos);
        cv::Mat tmp = newMinority.row(samples);
        minority.row(pos).copyTo(tmp);
        samples++;
      }
    }
    minority = newMinority;
  }

  cv::Mat synthetic(amountSmote, attributes, CV_32FC1);
  cv::Mat neighbors(minoritySamples, nearestNeighbors, CV_32FC1);

  /* Compute all the neighbors for the minority class */
  computeNeighbors(minority, nearestNeighbors, &neighbors);

  /* For each sample, generate it(s) synthetic(s) sample(s) */
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, minority.rows-1);

  while (amountSmote > 0) {
    populate(minority, neighbors, &synthetic, &index, 1, dis(gen), nearestNeighbors);
    amountSmote--;
  }
  return synthetic;
}
