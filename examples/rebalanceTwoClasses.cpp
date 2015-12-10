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

  /* If descriptor ==  CCV, threshold is required */
  if (d == 3)
    return PerformFeatureExtraction(dir, features, d, 64, 1, 0, paramCCV, 0, m, id);
  /* If descriptor ==  ACC, distances are required */
  else if (d == 5)
    return PerformFeatureExtraction(dir, features, d, 64, 1, 0, paramACC, 0, m, id);
  else
    return PerformFeatureExtraction(dir, features, d, 64, 1, 0, parameters, 0, m, id);
}

vector<vector<double> > perform(string descriptorFile, int repeat, double prob, string csv) {
  Classifier c;
  int minoritySize, numClasses;
  vector<vector<double> > fscore;
  // Read the feature vectors
  vector<Classes> data = ReadFeaturesFromFile(descriptorFile);
  numClasses = data.size();
  if (numClasses != 0){
    cout << "---------------------------------------------------------------------------------------" << endl;
    cout << "Features vectors file: " << descriptorFile.c_str() << endl;
    cout << "---------------------------------------------------------------------------------------" << endl;
    c.findSmallerClass(data, &minoritySize);
    fscore = c.classify(prob, repeat, data, csv.c_str(), minoritySize);
    data.clear();
  }
  return fscore;
}


void ShufflePaths(vector< vector<string>> *pathsForEachClass){

  int i, j, k, numImages;
  string aux;

  for (i = 0; i < (*pathsForEachClass).size(); i++) {
    numImages = (*pathsForEachClass)[i].size();
    for (j = numImages-1; j > 1; j--) {
      k = rand()%j;
      // Swap with k position
      aux = (*pathsForEachClass)[i][j];
      (*pathsForEachClass)[i][j] = (*pathsForEachClass)[i][k];
      (*pathsForEachClass)[i][k] = aux;
    }
  }
}


// Arrange original images in k folds each, creating k fold_i.txt files
vector< vector< vector<string>>> SeparateInFolds(string images_directory, int k) {

  vector< vector<string>> pathsForEachClass;
  vector<string> pathsInThisClass;
  vector< vector< vector<string>>> foldsForEachClass;
  vector< vector<string>> foldsInThisClass;
  vector<string> auxFold;
  vector<int> numFoldsInThisClass, num_images_class;

  int i, j, fold, qtdImgTotal, numClasses, numImages, treino;
  int resto, imgsInThisFold, imgTotal, next;
  Mat labels, trainTest, isGenerated, img;

  numClasses = qtdArquivos(images_directory);
  qtdImgTotal = NumberImagesInDataset(images_directory, numClasses, &num_images_class);
  trainTest = Mat::zeros(qtdImgTotal, 1, CV_32S);
  isGenerated= Mat::zeros(qtdImgTotal, 1, CV_32S);

  for (i = 0; i < numClasses; i++) {
    NumberImgInClass(images_directory, i, &numImages, &treino);
    imgTotal = 0;
    for (j = 0; j < numImages; j++) {
      // Find this image in the class and save the path
      img = FindImgInClass(images_directory, i, j, imgTotal, treino, &trainTest,  &pathsInThisClass, &isGenerated);
      if (!img.empty()) {
        imgTotal++;
      }
    }
    pathsForEachClass.push_back(pathsInThisClass);
    pathsInThisClass.clear();
    numFoldsInThisClass.push_back(numImages/k);
  }

  ShufflePaths(&pathsForEachClass);
  for (i = 0; i < numClasses; i++) {
    resto = 0;
    next = 0;

    for (fold = 0; fold < k; fold++) {

      if (num_images_class[i] % k > resto) {
        numFoldsInThisClass[i] = numFoldsInThisClass[i]+1;
        resto++;
      }
      imgsInThisFold = numFoldsInThisClass[i];
      for (j = 0; j < imgsInThisFold; j++) {
        auxFold.push_back(pathsForEachClass[i][next]);
        next++;
      }
      foldsInThisClass.push_back(auxFold);
      auxFold.clear();
    }
    foldsForEachClass.push_back(foldsInThisClass);
    foldsInThisClass.clear();
  }

  return foldsForEachClass;
}

int main(int argc, char const *argv[]) {
  Classifier c;
  Size size;
  int m, d, operation, indexDescriptor, i, numImages;
  int repeatRebalance, repeat, count_diff, prev, qtd_classes, treino;
  double prob = 0.5;
  string name, method, newDir, baseDir, featuresDir, csvOriginal, csvSmote;
  string csvRebalance, analysisDir, csvDesbalanced, directory, str, op;
  string images_directory, imbalancedDescriptor, originalDescriptor;
  string descSmote, smoteDescriptor;
  vector<Classes> imbalancedData;
  Artificial a;

  repeatRebalance = 1;
  if (argc < 3) {
    cout << "\nUsage: ./rebalanceTest (0) (1) (2) (3) (4)\n " << endl;
    cout << "\t(0) Directory to place tests\n" << endl;
    cout << "\t(1) Image Directory\n" << endl;
    exit(-1);
  }
  newDir = string(argv[1]);
  baseDir = string(argv[2]);
  if (argc == 4) repeatRebalance = atoi(argv[3]);

  analysisDir = newDir+"/analysis/";
  featuresDir = newDir+"/features/";

  srand(1);
  //  Available methods:
  // Description {"BIC", "GCH", "CCV", "Haralick6", "ACC", "LBP", "HOG", "Contour"}
  // Quantization {"Intensity", "Luminance", "Gleam", "MSB", "MSB-Modified", "BGR", "HSV"}

  // vector <int> descriptors {1, 6, 7};
  // vector <int> quant {1, 3, 2};

  vector <int> descriptors {1};
  vector <int> quant {1};

  double factor = 0.1;

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

  // Arrange original images in k folds each, creating k fold_i.txt files
  vector< vector< vector<string>>> folds;
  folds = SeparateInFolds(baseDir+"/", 10);
  for (i = 0; i < folds.size(); i++) {
    cout << "Classe " << i << endl;
    for (int x = 0; x < folds[i].size(); x++) {
      cout << "\tFold " << x << endl;
      for (int j = 0; j < folds[i][x].size(); j++) {
        cout << "\t\t " << folds[i][x][j] << endl;
      }
    }
  }

  // int testing_fold, training_fold;
  // for (i = 0; i < folds.size(); i++) {
  //   cout << "Classe " << i << endl;
  //   for (testing_fold = 0; testing_fold < folds[i].size(); testing_fold++) {
  //     cout << "testing_fold " << testing_fold << endl;
  //
  //     for (training_fold = 0; training_fold < folds[i].size(); training_fold++) {
  //       if (training_fold != testing_fold) {
  //         cout << "training_fold "<< training_fold << endl;
  //         // for (int j = 0; j < folds[i][x].size(); j++) {
  //         //   cout << "\t\t " << folds[i][x][j] << endl;
  //         // }
  //       }
  //     }
  //   }
  // }

  cout << "\n\n------------------------------------------------------------------------------------" << endl;
  cout << "Select a single fold to train the  minority class:" << endl;
  cout << "---------------------------------------------------------------------------------------" << endl;
  images_directory = RemoveSamples(baseDir, newDir, prob, factor);

  // For each rebalancing operation
  for (operation = 0; operation <= 0; operation++){

    vector<String> allRebalanced;
    stringstream operationstr;
    operationstr << operation;
    op = operationstr.str();

    /* Generate Artificial Images */
    repeatRebalance = 1;
    for (repeat = 0; repeat < repeatRebalance; repeat++) {
      stringstream repeatStr;
      repeatStr << repeat;
      string newDirectory = newDir+"/Artificial/"+op+"-Rebalanced"+repeatStr.str();
      string dirRebalanced = a.generate(images_directory, newDirectory, operation);
      allRebalanced.push_back(dirRebalanced);
    }

    for (indexDescriptor = 0; indexDescriptor < (int)descriptors.size(); indexDescriptor++){
      d = descriptors[indexDescriptor];
      m = quant[indexDescriptor];

      // Feature extraction from images
      if (count_diff == 0) {
        cout << "Classification using original data" << endl;
        string originalDescriptor = description(baseDir, featuresDir, d, m, "original");
        csvOriginal = analysisDir+op+"-original_"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
        perform(originalDescriptor, 10, prob, csvOriginal);
      }

      cout << "Classification using desbalanced data" << endl;
      string imbalancedDescriptor = description(images_directory, featuresDir, d, m, "desbalanced");
      csvDesbalanced = analysisDir+op+"-desbalanced_"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
      perform(imbalancedDescriptor, 1, prob, csvDesbalanced);

      for (i = 0; i < (int)allRebalanced.size(); i++) {
        // Generate Synthetic SMOTE samples
        imbalancedData = ReadFeaturesFromFile(imbalancedDescriptor);
        descSmote = newDir+"/features/"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
        smoteDescriptor = PerformSmote(imbalancedData, operation, descSmote);
        cout << "Classification using SMOTE" << endl;
        csvSmote = analysisDir+op+"-smote_"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
        perform(smoteDescriptor, 1, prob, csvSmote);
        imbalancedData.clear();

        cout << "Classification using rebalanced data" << endl;
        featuresDir = allRebalanced[i]+"/../features/";
        string artificialDescriptor = description(allRebalanced[i], featuresDir, d, m, "artificial");
        csvRebalance = analysisDir+op+"-artificial_"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
        perform(artificialDescriptor, 1, prob, csvRebalance);
      }
    }
    allRebalanced.clear();
  }

  cout << "Done." << endl;
  return 0;
}
