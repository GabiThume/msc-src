/*
Copyright (c) 2015, All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other Materials provided with the
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

int main(int argc, char const *argv[]) {
  Classifier c;
  cv::Size size;
  int m, d, operation, indexDescriptor, i, numImages;
  int repeatRebalance, repeat, count_diff, prev, qtd_classes, treino;
  double prob = 0.5;
  std::string name, method, newDir, baseDir, featuresDir, csvOriginal, csvSmote;
  std::string csvRebalance, analysisDir, csvDesbalanced, directory, str, op;
  std::string images_directory, imbalancedDescriptor, originalDescriptor;
  std::string descSmote, smoteDescriptor, dirRebalanced, artificialDescriptor;
  std::vector<ImageClass> imbalancedData;
  Artificial a;
  int minorityClass = 1, thisClass, j, x;
  std::vector<int> testing_fold, majority_fold, minority_fold;

  int k = 10;
  repeatRebalance = 1;
  if (argc < 3) {
    std::cout << "\nUsage: ./rebalanceTest (0) (1) (2) (3) (4)\n " << std::endl;
    std::cout << "\t(0) Directory to place tests\n" << std::endl;
    std::cout << "\t(1) Image Directory\n" << std::endl;
    exit(-1);
  }
  newDir = std::string(argv[1]);
  baseDir = std::string(argv[2]);
  if (argc == 4) repeatRebalance = atoi(argv[3]);

  analysisDir = newDir+"/analysis/";
  featuresDir = newDir+"/features/";

  srand(1);
  //  Available methods:
  // Description {"BIC", "GCH", "CCV", "Haralick6", "ACC", "LBP", "HOG", "Contour"}
  // Quantization {"Intensity", "Luminance", "Gleam", "MSB", "MSB-Modified", "BGR", "HSV"}

  // std::vector <int> descriptors {1, 6, 7};
  // std::vector <int> quant {1, 3, 2};

  std::vector <int> descriptors {1};
  std::vector <int> quant {1};

  double factor = 0.1;

  // // Check how many classes and images there are
  // qtd_classes = qtdArquivos(baseDir+"/");
  // if (qtd_classes < 2) {
  //   std::cout << "Error. There is less than two classes in " << baseDir << std::endl;
  //   exit(-1);
  // }
  //
  // NumberImgInClass(baseDir, 0, &prev, &treino);
  // count_diff = 0;
  // for (i = 1; i < qtd_classes; i++) {
  //   NumberImgInClass(baseDir, i, &numImages, &treino);
  //   count_diff = count_diff + abs(numImages - prev);
  //   prev = numImages;
  // }

  d = 1; m = 1;
  operation = 1;

  // std::cout << "Classification using original data" << std::endl;
  Rebalance r;
  r.extractor.ccvThreshold = 25;
  r.extractor.accDistances = {1, 3, 5, 7};
  r.extractor.normalization = 0;
  r.extractor.resizeFactor = 1.0;

  r.readImageDirectory(baseDir);
  r.performFeatureExtraction(d, m);

  // // perform(originalDescriptor, 10, prob, csvOriginal);
  // std::vector<ImageClass> data = ReadFeaturesFromFile(originalDescriptor);
  //
  // // Arrange original images in k folds each, creating k fold_i.txt files
  // SeparateInFolds(&data, k);
  //
  // for (i = 0; i < k; i++) {
  //   for (j = 0; j < k; j++) {
  //     if (j != i) {
  //       for (thisClass = 0; thisClass < data.size(); thisClass++) {
  //         data[thisClass].testing_fold.push_back(i);
  //         if (thisClass == minorityClass) {
  //           data[thisClass].training_fold.push_back(j);
  //         }
  //         else {
  //           for (x = 0; x < k; x++) {
  //             if (x != i) {
  //               data[thisClass].training_fold.push_back(x);
  //             }
  //           }
  //         }
  //       }
  //       std::vector<ImageClass> generated = a.generateImagesFromData(data, newDir+"/Artificial/", operation);
  //       featuresDir = newDir+"/Artificial/"+"/features/";
  //       artificialDescriptor = PerformFeatureExtraction(generated, featuresDir, d, 64, 1, 0, paramCCV, 0, m, "artificial");
  //       // artificialDescriptor = description(dirRebalanced, featuresDir, d, m, "artificial");
  //       // csvRebalance = analysisDir+op+"-artificial_"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
  //       // perform(artificialDescriptor, 1, prob, csvRebalance);
  //
  //       exit(1);
  //     }
  //   }
  // }

  // std::cout << "\n\n------------------------------------------------------------------------------------" << std::endl;
  // std::cout << "Select a single fold to train the  minority class:" << std::endl;
  // std::cout << "---------------------------------------------------------------------------------------" << std::endl;
  // images_directory = RemoveSamples(baseDir, newDir, prob, factor);

  // // For each rebalancing operation
  // for (operation = 0; operation <= 0; operation++){
  //
  //   std::vector<String> allRebalanced;
  //   std::stringstream operationstr;
  //   operationstr << operation;
  //   op = operationstr.str();
  //
  //   /* Generate Artificial Images */
  //   repeatRebalance = 1;
  //   for (repeat = 0; repeat < repeatRebalance; repeat++) {
  //     std::stringstream repeatStr;
  //     repeatStr << repeat;
  //     std::string newDirectory = newDir+"/Artificial/"+op+"-Rebalanced"+repeatStr.str();
  //     std::string dirRebalanced = a.generate(images_directory, newDirectory, operation);
  //     allRebalanced.push_back(dirRebalanced);
  //   }
  //
  //   for (indexDescriptor = 0; indexDescriptor < (int)descriptors.size(); indexDescriptor++){
  //     d = descriptors[indexDescriptor];
  //     m = quant[indexDescriptor];
  //
  //     // Feature extraction from images
  //     if (count_diff == 0) {
  //       std::cout << "Classification using original data" << std::endl;
  //       std::string originalDescriptor = description(baseDir, featuresDir, d, m, "original");
  //       csvOriginal = analysisDir+op+"-original_"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
  //       perform(originalDescriptor, 10, prob, csvOriginal);
  //     }
  //
  //     std::cout << "Classification using desbalanced data" << std::endl;
  //     std::string imbalancedDescriptor = description(images_directory, featuresDir, d, m, "desbalanced");
  //     csvDesbalanced = analysisDir+op+"-desbalanced_"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
  //     perform(imbalancedDescriptor, 1, prob, csvDesbalanced);
  //
  //     for (i = 0; i < (int)allRebalanced.size(); i++) {
  //       // Generate Synthetic SMOTE samples
  //       imbalancedData = ReadFeaturesFromFile(imbalancedDescriptor);
  //       descSmote = newDir+"/features/"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
  //       smoteDescriptor = PerformSmote(imbalancedData, operation, descSmote);
  //       std::cout << "Classification using SMOTE" << std::endl;
  //       csvSmote = analysisDir+op+"-smote_"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
  //       perform(smoteDescriptor, 1, prob, csvSmote);
  //       imbalancedData.clear();
  //
  //       std::cout << "Classification using rebalanced data" << std::endl;
  //       featuresDir = allRebalanced[i]+"/../features/";
  //       std::string artificialDescriptor = description(allRebalanced[i], featuresDir, d, m, "artificial");
  //       csvRebalance = analysisDir+op+"-artificial_"+descriptorMethod[d-1]+"_"+quantizationMethod[m-1]+"_";
  //       perform(artificialDescriptor, 1, prob, csvRebalance);
  //     }
  //   }
  //   allRebalanced.clear();
  // }

  std::cout << "Done." << std::endl;
  return 0;
}
