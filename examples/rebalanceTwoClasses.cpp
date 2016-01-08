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
  int m, d, operation, i, experiment, repeatRebalance, repeat, minorityClass, j, x;
  std::string newDir, baseDir, featuresDir, analysisDir, generationDir, str;
  std::vector<ImageClass> imbalancedData;
  Artificial a;
  DIR *dir = NULL;
  std::vector<int> testing_fold, majority_fold, minority_fold;
  std::vector<ImageClass>::iterator itClass;

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

  srand(time(NULL));
  //  Available methods:
  // Description {"BIC", "GCH", "CCV", "Haralick6", "ACC", "LBP", "HOG", "Contour"}
  // Quantization {"Intensity", "Luminance", "Gleam", "MSB", "MSB-Modified", "BGR", "HSV"}

  // std::vector <int> descriptors {1, 6, 7};
  // std::vector <int> quant {1, 3, 2};
  d = 1; m = 1;
  operation = 1;

  Rebalance r;
  r.extractor.ccvThreshold = 25;
  r.extractor.accDistances = {1, 3, 5, 7};
  r.extractor.normalization = 0;
  r.extractor.resizeFactor = 1.0;
  r.extractor.numColors = 64;

  r.readImageDirectory(baseDir);
  r.performFeatureExtraction(d, m);
  // Arrange original images in k folds each
  r.separateInFolds(k);
  r.writeFeatures(newDir+"/original-folds");

  // minorityClass = r.data.smallerTrainingClass();
  minorityClass = 1;
  experiment = 0;
  for (i = 0; i < k; i++) {
    for (j = 0; j < k; j++) {
      if (j != i) {
        for (itClass = r.data.classes.begin(); itClass != r.data.classes.end(); ++itClass) {
          itClass->testing_fold.clear();
          itClass->training_fold.clear();
          itClass->generated_fold.clear();
          itClass->smote_fold.clear();
          itClass->testing_fold.push_back(i);
          if (itClass->id == minorityClass) {
            itClass->training_fold.push_back(j);
          }
          else {
            for (x = 0; x < k; x++) {
              if (x != i) {
                itClass->training_fold.push_back(x);
              }
            }
          }
        }
        for (itClass = r.data.classes.begin(); itClass != r.data.classes.end(); ++itClass) {
          std::cout << "Class " << itClass->id << std::endl;
          for (x = 0; x < itClass->testing_fold.size(); x++) {
            std::cout << "\t Testing fold " << itClass->testing_fold[x] << std::endl;
          }
          for (x = 0; x < itClass->training_fold.size(); x++) {
            std::cout << "\t Training fold " << itClass->training_fold[x] << std::endl;
          }
        }

        generationDir = newDir+"/artificial/experiment-"+to_string(experiment)+"/operation-"+to_string(operation)+"/";
        featuresDir = generationDir+"features/";
        analysisDir = generationDir+"analysis/";

        dir = opendir(generationDir.c_str());
      	if (!dir) {
      		// str = "rm -f -r "+generationDir+"/*;";
      		str = "mkdir -p "+generationDir+";";
          str += "mkdir -p "+featuresDir+";";
          str += "mkdir -p "+analysisDir+";";
      		system(str.c_str());
      		dir = opendir(generationDir.c_str());
      		if (!dir) {
      			std::cout << "Error! Directory " << generationDir;
      			std::cout << " can't be created. " << std::endl;
      			exit(1);
      		}
      	}
      	closedir(dir);


        // Classify original folds
        r.writeFeatures(featuresDir+"original");
        c.classify(0.5,  2, r.data, analysisDir+"original", 1);

        // Perform SMOTE subsampling
        r.performSmote(&r.data, operation);
        for (itClass = r.data.classes.begin(); itClass != r.data.classes.end(); ++itClass) {
          for (x = 0; x < itClass->smote_fold.size(); x++) {
            itClass->training_fold.push_back(itClass->smote_fold[x]);
          }
        }
        r.writeFeatures(featuresDir+"smote");
        c.classify(0.5,  2, r.data, analysisDir+"smote", 1);
        for (itClass = r.data.classes.begin(); itClass != r.data.classes.end(); ++itClass) {
          for (x = 0; x < itClass->smote_fold.size(); x++) {
            itClass->training_fold.erase(std::remove(itClass->training_fold.begin(),
                                                    itClass->training_fold.end(),
                                                    itClass->smote_fold[x]),
                                        itClass->training_fold.end());
            int current_fold = itClass->smote_fold[x];
            itClass->images.erase(std::remove_if(itClass->images.begin(),
                                                itClass->images.end(),
                                                [current_fold] (Image y){
                                                  return (y.fold == current_fold);
                                                }),
                                  itClass->images.end());
          }
        }

        // Perform images artificial generation
        a.generateImagesFromData(&r.data,
                                generationDir+"/images/",
                                operation);
          for (itClass = r.data.classes.begin(); itClass != r.data.classes.end(); ++itClass) {
            for (x = 0; x < itClass->generated_fold.size(); x++) {
              itClass->training_fold.push_back(itClass->generated_fold[x]);
            }
          }
        r.performFeatureExtraction(d, m);
        r.writeFeatures(featuresDir+"artificial");
        c.classify(0.5,  2, r.data, analysisDir+"artificial", 1);
        for (itClass = r.data.classes.begin(); itClass != r.data.classes.end(); ++itClass) {
          for (x = 0; x < itClass->generated_fold.size(); x++) {
            itClass->training_fold.erase(std::remove(itClass->training_fold.begin(),
                                                    itClass->training_fold.end(),
                                                    itClass->generated_fold[x]),
                                        itClass->training_fold.end());
            int current_fold = itClass->generated_fold[x];
            itClass->images.erase(std::remove_if(itClass->images.begin(),
                                                itClass->images.end(),
                                                [current_fold] (Image y) {
                                                  return (y.fold == current_fold);
                                                }),
                                  itClass->images.end());
          }
        }
        experiment++;
        // exit(0);
      }
    }
  }

  // std::vector<ImageClass> data = ReadFeaturesFromFile(originalDescriptor);
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

  std::cout << "-------------------------------------------------" << std::endl;
  std::cout << "Rebalance done." << std::endl;
  return 0;
}
