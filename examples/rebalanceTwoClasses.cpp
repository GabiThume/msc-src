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


string description(string dir, string features, int d, int m, string id) {

  vector<int> paramCCV = {25}, paramACC = {1, 3, 5, 7}, parameters;

  // string descriptorMethod[8] = {"BIC", "GCH", "CCV", "Haralick6", "ACC", "LBP", "HOG", "Contour"};
  // string quantizationMethod[4] = {"Intensity", "Luminance", "Gleam", "MSB"};
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
  Size size;
  Artificial a;
  ofstream csvFile;
  stringstream globalFactor;
  int numClasses, m, d, operation, indexDescriptor, minoritySize, i, numImages;
  int repeatRebalance = 20, repeat, count_diff, prev, qtd_classes, treino;
  double prob = 0.5;
  string nameFile, name, nameDir, descriptorName, method, newDir, baseDir, featuresDir;
  string csvOriginal, csvSmote, csvRebalance, analysisDir, csvDesbalanced;
  string directory, str, bestDir, op, descSmote, smoteDescriptor, fileDescriptor;
  string images_directory, originalDescriptor;
  vector<Classes> imbalancedData, artificialData, originalData, rebalancedData;
  vector<int> objperClass;
  vector<vector<double> > rebalancedFscore, desbalancedFscore;
  srand(time(NULL));

  if (argc != 3){
    cout << "\nUsage: ./rebalanceTest (0) (1) (2) (3) (4)\n " << endl;
    cout << "\t(0) Directory to place tests\n" << endl;
    cout << "\t(1) Image Directory\n" << endl;
    exit(-1);
  }
  newDir = string(argv[1]);
  baseDir = string(argv[2]);

  /*  Available
  descriptorMethod: {"BIC", "GCH", "CCV", "Haralick6", "ACC", "LBP", "HOG", "Contour", "Fisher"}
  Quantization quantizationMethod: {"Intensity", "Luminance", "Gleam", "MSB"}
  */
  vector <int> descriptors {1, 6, 7};
  vector <int> quant {1, 3, 2};

  double factor = 1.6;

  // Check how many classes and images there are
  qtd_classes = qtdArquivos(baseDir+"/");
  if (qtd_classes < 2) {
    cout << "Error. There is less than two classes in " << baseDir << endl;
    exit(-1);
  }

  NumberImgInClass(baseDir, 0, &prev, &treino);
  count_diff = 0;
  for (i = 1; i < qtd_classes; i++) {
    NumberImgInClass(baseDir, i, &numImages, &treino);
    count_diff = count_diff + abs(numImages - prev);
    prev = numImages;
  }

  /* Desbalancing Data */
  cout << "\n\n------------------------------------------------------------------------------------" << endl;
  cout << "Divide the number of original samples to create a minority class:" << endl;
  cout << "---------------------------------------------------------------------------------------" << endl;
  images_directory = RemoveSamples(baseDir, newDir, 0.5, factor);

  // For each rebalancing operation
  for (operation = 0; operation <= 10; operation++){

    vector<String> allRebalanced;
    stringstream operationstr;
    operationstr << operation;
    op = operationstr.str();
    /* Generate Artificial Images */
    // string newDirectory = newDir+"/Rebalanced-"+op;
    // string dirRebalanced = a.generate(dirImbalanced, newDirectory, operation);
    for (repeat = 0; repeat < repeatRebalance; repeat++) {
      stringstream repeatStr;
      repeatStr << repeat;
      string newDirectory = newDir+"/"+op+"-Rebalanced"+repeatStr.str();
      string dirRebalanced = a.generate(images_directory, newDirectory, operation);
      allRebalanced.push_back(dirRebalanced);
    }

    for (indexDescriptor = 0; indexDescriptor < (int)descriptors.size(); indexDescriptor++){
      d = descriptors[indexDescriptor];
      m = quant[indexDescriptor];
      // for (m = 1; m <= 5; m++){
        csvOriginal = newDir+"/analysis/"+op+"-original_"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
        csvDesbalanced = newDir+"/analysis/"+op+"-desbalanced_"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
        csvSmote = newDir+"/analysis/"+op+"-smote_"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
        csvRebalance = newDir+"/analysis/"+op+"-artificial_"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
        featuresDir = newDir+"/features/";

        // Feature extraction from images
        if (count_diff == 0) {
          originalDescriptor = description(baseDir, featuresDir, d, m, "original");
          // Read the feature vectors
          originalData = ReadFeaturesFromFile(originalDescriptor);
          numClasses = originalData.size();
          if (numClasses != 0){
            cout << "---------------------------------------------------------------------------------------" << endl;
            cout << "Classification using original data" << endl;
            cout << "Features vectors file: " << originalDescriptor.c_str() << endl;
            cout << "---------------------------------------------------------------------------------------" << endl;
            c.findSmallerClass(originalData, &minoritySize);
            c.classify(prob, 10, originalData, csvOriginal.c_str(), minoritySize);
            originalData.clear();
          }
        }
        // Feature extraction from images
        originalDescriptor = description(images_directory, featuresDir, d, m, "desbalanced");
        // Read the feature vectors
        imbalancedData = ReadFeaturesFromFile(originalDescriptor);
        numClasses = imbalancedData.size();
        if (numClasses != 0){
          cout << "---------------------------------------------------------------------------------------" << endl;
          cout << "Classification using desbalanced data" << endl;
          cout << "Features vectors file: " << originalDescriptor.c_str() << endl;
          cout << "---------------------------------------------------------------------------------------" << endl;
          c.findSmallerClass(imbalancedData, &minoritySize);
          desbalancedFscore = c.classify(prob, 1, imbalancedData, csvDesbalanced.c_str(), minoritySize);
          imbalancedData.clear();
        }

        for (i = 0; i < (int)allRebalanced.size(); i++) {
          cout << "allRebalanced " << allRebalanced[i] << endl;
          // Generate Synthetic SMOTE samples
          imbalancedData = ReadFeaturesFromFile(originalDescriptor);
          descSmote = newDir+"/features/"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
          cout << descSmote << endl;
          smoteDescriptor = PerformSmote(imbalancedData, operation, descSmote);
          cout << smoteDescriptor << endl;
          rebalancedData = ReadFeaturesFromFile(smoteDescriptor);
          if (rebalancedData.size() != 0){
            cout << "---------------------------------------------------------------------------------------" << endl;
            cout << "Classification using SMOTE" << endl;
            cout << "Features vectors file: " << smoteDescriptor.c_str() << endl;
            cout << "---------------------------------------------------------------------------------------" << endl;
            c.findSmallerClass(rebalancedData, &minoritySize);
            cout << csvSmote << endl;
            c.classify(prob, 1, rebalancedData, csvSmote.c_str(), minoritySize);
            rebalancedData.clear();
          }

          featuresDir = allRebalanced[i]+"/../features/";
          fileDescriptor = description(allRebalanced[i], featuresDir, d, m, "artificial");
          artificialData = ReadFeaturesFromFile(fileDescriptor);
          if (artificialData.size() != 0){
            cout << "---------------------------------------------------------------------------------------" << endl;
            cout << "Classification using rebalanced data" << endl;
            cout << "Features vectors file: " << fileDescriptor.c_str() << endl;
            cout << "---------------------------------------------------------------------------------------" << endl;
            c.findSmallerClass(artificialData, &minoritySize);
            rebalancedFscore = c.classify(prob, 1, artificialData, csvRebalance.c_str(), minoritySize);
            artificialData.clear();
          }
        }
      // }
    }
    allRebalanced.clear();
  }

  cout << "Done." << endl;
  return 0;
}
