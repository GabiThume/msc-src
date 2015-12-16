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

#include "rebalanceTest.h"
#include "utils/rebalance.h"


std::string description(std::string dir, std::string features, int d, int m, std::string id) {

  std::vector<int> paramCCV = {25}, paramACC = {1, 3, 5, 7}, parameters;

  // std::string descriptorMethod[8] = {"BIC", "GCH", "CCV", "Haralick6", "ACC", "LBP", "HOG", "Contour"};
  // std::string quantizationMethod[4] = {"Intensity", "Luminance", "Gleam", "MSB"};
  /* If descriptor ==  CCV, threshold is required */
  if (d == 3)
    return PerformFeatureExtraction(dir, features, d, 64, 1, 0, paramCCV, 0, m, id);
  /* If descriptor ==  ACC, distances are required */
  else if (d == 5)
    return PerformFeatureExtraction(dir, features, d, 64, 1, 0, paramACC, 0, m, id);
  else
    return PerformFeatureExtraction(dir, features, d, 64, 1, 0, parameters, 0, m, id);
}


int main(int argc, char const *argv[]) {
  Classifier c;
  cv::Size size;
  std::ofstream csvFile;
  std::stringstream globalFactor;
  int numClasses, m, d, indexDescriptor, minoritySize;
  double prob = 0.5;
  std::string nameFile, name, nameDir, descriptorName, method, newDir, baseDir, featuresDir;
  std::string csvOriginal, csvSmote, csvRebalance, analysisDir, csvDesbalanced;
  std::string directory, str, bestDir, op, descSmote, smoteDescriptor, fileDescriptor;
  std::string images_directory, originalDescriptor;
  std::vector<ImageClass> imbalancedData, artificialData, originalData, rebalancedData;
  std::vector<int> objperClass;
  std::vector<std::vector<double> > rebalancedFscore, desbalancedFscore;
  srand(time(0));

  if (argc != 3){
    std::cout << "\nUsage: ./smoteTest (0) (1)\n " << std::endl;
    std::cout << "\t(0) Directory to place tests\n" << std::endl;
    std::cout << "\t(1) Image Directory\n" << std::endl;
    exit(-1);
  }
  newDir = std::string(argv[1]);
  baseDir = std::string(argv[2]);

  /*  Available
  descriptorMethod: {"BIC", "GCH", "CCV", "Haralick6", "ACC", "LBP", "HOG", "Contour", "Fisher"}
  Quantization quantizationMethod: {"Intensity", "Luminance", "Gleam", "MSB"}
  */
  std::vector <int> descriptors {1, 3, 6, 7, 8};

  for (indexDescriptor = 0; indexDescriptor < (int)descriptors.size(); indexDescriptor++){
    d = descriptors[indexDescriptor];
    for (m = 1; m <= 5; m++){
      csvOriginal = newDir+"/analysis/"+op+"-original_"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
      csvSmote = newDir+"/analysis/"+op+"-smote_"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
      featuresDir = newDir+"/features/";

      // Feature extraction from images
      originalDescriptor = description(baseDir, featuresDir, d, m, "original");
      // Read the feature vectors
      originalData = ReadFeaturesFromFile(originalDescriptor);
      numClasses = originalData.size();
      if (numClasses != 0){
        std::cout << "---------------------------------------------------------------------------------------" << std::endl;
        std::cout << "Classification using original data" << std::endl;
        std::cout << "Features vectors file: " << originalDescriptor.c_str() << std::endl;
        std::cout << "---------------------------------------------------------------------------------------" << std::endl;
        c.findSmallerClass(originalData, &minoritySize);
        c.classify(prob, 10, originalData, csvOriginal.c_str(), minoritySize);
        originalData.clear();
      }

      // Generate Synthetic SMOTE samples
      imbalancedData = ReadFeaturesFromFile(originalDescriptor);
      descSmote = newDir+"/features/"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
      smoteDescriptor = PerformSmote(imbalancedData, 1, descSmote);
      rebalancedData = ReadFeaturesFromFile(smoteDescriptor);
      if (rebalancedData.size() != 0){
        std::cout << "---------------------------------------------------------------------------------------" << std::endl;
        std::cout << "Classification using SMOTE" << std::endl;
        std::cout << "Features vectors file: " << smoteDescriptor.c_str() << std::endl;
        std::cout << "---------------------------------------------------------------------------------------" << std::endl;
        c.findSmallerClass(rebalancedData, &minoritySize);
        c.classify(prob, 10, rebalancedData, csvSmote.c_str(), minoritySize);
        rebalancedData.clear();
      }
    }
  }

  std::cout << "Done." << std::endl;
  return 0;
}
