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


void ShuffleImages(vector<Image> *img){

  int i, k, numImages;
  Image aux;

  numImages = (*img).size();
  for (i = numImages-1; i > 1; i--) {
    k = rand()%i;
    // Swap with k position
    aux = (*img)[i];
    (*img)[i] = (*img)[k];
    (*img)[k] = aux;
  }
}


// Arrange original images in k folds each, creating k fold_i.txt files
void SeparateInFolds(vector<Classes> *original_data, int k) {

  int i, j, fold, numClasses, numImages, resto, imgsInThisFold, imgTotal, next;
  Mat labels, trainTest, isGenerated, img;

  numClasses = (*original_data).size();
  for (i = 0; i < numClasses; i++) {
    resto = 0; next = 0;

    ShuffleImages(&(*original_data)[i].images);

    numImages = (*original_data)[i].images.size();
    for (fold = 0; fold < k; fold++) {
      // If the number of images per fold is less than the total, increase each extra image in a fold
      imgsInThisFold = floor(numImages/k);
      if (numImages % k > resto) {
        imgsInThisFold++;
        resto++;
      }
      // For each image requested for this fold, add the fold indication
      for (j = 0; j < imgsInThisFold; j++) {
        (*original_data)[i].images[next].fold = fold;
        next++;
      }
    }
  }
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
  int minorityClass = 1, thisClass, j, x;
  vector<int> testing_fold, majority_fold, minority_fold;

  int k = 10;
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

  d = 1; m = 1;

  cout << "Classification using original data" << endl;
  originalDescriptor = description(baseDir, featuresDir, d, m, "original");
  csvOriginal = analysisDir+op+"-original_"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
  // perform(originalDescriptor, 10, prob, csvOriginal);
  vector<Classes> data = ReadFeaturesFromFile(originalDescriptor);

  // Arrange original images in k folds each, creating k fold_i.txt files
  SeparateInFolds(&data, k);

  for (i = 0; i < k; i++) {
    for (j = 0; j < k; j++) {
      if (j != i) {
        for (thisClass = 0; thisClass < data.size(); thisClass++) {
          data[thisClass].testing_fold.push_back(i);
          if (thisClass == minorityClass) {
            data[thisClass].training_fold.push_back(j);
          }
          else {
            for (x = 0; x < k; x++) {
              if (x != i) {
                data[thisClass].training_fold.push_back(x);
              }
            }
          }
        }
        string dirRebalanced = a.generateImagesFromData(data, newDir+"/Artificial/", operation);
        exit(1);
      }
    }
  }

  // cout << "\n\n------------------------------------------------------------------------------------" << endl;
  // cout << "Select a single fold to train the  minority class:" << endl;
  // cout << "---------------------------------------------------------------------------------------" << endl;
  // images_directory = RemoveSamples(baseDir, newDir, prob, factor);

  // // For each rebalancing operation
  // for (operation = 0; operation <= 0; operation++){
  //
  //   vector<String> allRebalanced;
  //   stringstream operationstr;
  //   operationstr << operation;
  //   op = operationstr.str();
  //
  //   /* Generate Artificial Images */
  //   repeatRebalance = 1;
  //   for (repeat = 0; repeat < repeatRebalance; repeat++) {
  //     stringstream repeatStr;
  //     repeatStr << repeat;
  //     string newDirectory = newDir+"/Artificial/"+op+"-Rebalanced"+repeatStr.str();
  //     string dirRebalanced = a.generate(images_directory, newDirectory, operation);
  //     allRebalanced.push_back(dirRebalanced);
  //   }
  //
  //   for (indexDescriptor = 0; indexDescriptor < (int)descriptors.size(); indexDescriptor++){
  //     d = descriptors[indexDescriptor];
  //     m = quant[indexDescriptor];
  //
  //     // Feature extraction from images
  //     if (count_diff == 0) {
  //       cout << "Classification using original data" << endl;
  //       string originalDescriptor = description(baseDir, featuresDir, d, m, "original");
  //       csvOriginal = analysisDir+op+"-original_"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
  //       perform(originalDescriptor, 10, prob, csvOriginal);
  //     }
  //
  //     cout << "Classification using desbalanced data" << endl;
  //     string imbalancedDescriptor = description(images_directory, featuresDir, d, m, "desbalanced");
  //     csvDesbalanced = analysisDir+op+"-desbalanced_"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
  //     perform(imbalancedDescriptor, 1, prob, csvDesbalanced);
  //
  //     for (i = 0; i < (int)allRebalanced.size(); i++) {
  //       // Generate Synthetic SMOTE samples
  //       imbalancedData = ReadFeaturesFromFile(imbalancedDescriptor);
  //       descSmote = newDir+"/features/"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
  //       smoteDescriptor = PerformSmote(imbalancedData, operation, descSmote);
  //       cout << "Classification using SMOTE" << endl;
  //       csvSmote = analysisDir+op+"-smote_"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
  //       perform(smoteDescriptor, 1, prob, csvSmote);
  //       imbalancedData.clear();
  //
  //       cout << "Classification using rebalanced data" << endl;
  //       featuresDir = allRebalanced[i]+"/../features/";
  //       string artificialDescriptor = description(allRebalanced[i], featuresDir, d, m, "artificial");
  //       csvRebalance = analysisDir+op+"-artificial_"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
  //       perform(artificialDescriptor, 1, prob, csvRebalance);
  //     }
  //   }
  //   allRebalanced.clear();
  // }

  cout << "Done." << endl;
  return 0;
}
