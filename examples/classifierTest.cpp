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

string desc(string dir, string features, int d, int m, string id){

    vector<int> paramCCV = {25};
    vector<int> paramACC = {1, 3, 5, 7};
    vector<int> parameters;
    /* If descriptor ==  CCV, threshold is required */
    if (d == 3)
        return PerformFeatureExtraction(dir, features, d, 64, 1, 1, paramCCV, 0, m, id);
    /* If descriptor ==  ACC, distances are required */
    else if (d == 5)
        return PerformFeatureExtraction(dir, features, d, 64, 1, 1, paramACC, 0, m, id);
    else
        return PerformFeatureExtraction(dir, features, d, 64, 1, 1, parameters, 0, m, id);
}

int main(int argc, char const *argv[]){

  Classifier c;
  int numClasses, m, d, indexDescriptor, minoritySize;
  double prob = 0.5;
  string newDir, baseDir, featuresDir, csvOriginal, directory, str, bestDir;
  vector<Classes> originalData;

  if (argc != 3){
    cout << "\nUsage: ./rebalanceTest (0) (1) (2) (3) (4)\n " << endl;
    cout << "\t(0) Directory to place tests\n" << endl;
    cout << "\t(1) Image Directory\n" << endl;
    cout << "\t(2) Features Directory\n" << endl;
    cout << "\t(3) Analysis Directory\n" << endl;
    cout << "\t./rebalanceTest Desbalanced/ Desbalanced/original/ Desbalanced/features/ Desbalanced/analysis/ 0\n" << endl;
    exit(-1);
  }
  newDir = string(argv[1]);
  baseDir = string(argv[2]);
  // featuresDir = string(argv[3]);
  // analysisDir = string(argv[4]);

  /*  Available
  descriptorMethod: {"BIC", "GCH", "CCV", "Haralick6", "ACC", "LBP", "HOG", "Contour", "Fisher"}
  Quantization quantizationMethod: {"Intensity", "Luminance", "Gleam", "MSB"}
  */
  vector <int> descriptors {1, 2, 3, 4, 5, 6, 7, 8};

  for (indexDescriptor = 0; indexDescriptor < (int)descriptors.size(); indexDescriptor++){
    d = descriptors[indexDescriptor];
    for (m = 1; m <= 5; m++){
      csvOriginal = newDir+"/analysis/original_"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
      featuresDir = newDir+"/features/";

      /* Feature extraction from images */
      string originalDescriptor = desc(baseDir, featuresDir, d, m, "original");
      /* Read the feature vectors */
      originalData = ReadFeaturesFromFile(originalDescriptor);
      numClasses = originalData.size();
      if (numClasses != 0){
        cout << "---------------------------------------------------------------------------------------" << endl;
        cout << "Classification using original data" << endl;
        cout << "Features vectors file: " << originalDescriptor.c_str() << endl;
        cout << "---------------------------------------------------------------------------------------" << endl;
        c.findSmallerClass(originalData, &minoritySize);
        c.classify(prob, 1, originalData, csvOriginal.c_str(), minoritySize);
        originalData.clear();
      }
    }
  }
  return 0;
}
