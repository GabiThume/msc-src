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
  int m, d, operation, i, experiment, j, x, indexDescriptor;
  std::string newDir, baseDir, featuresDir, analysisDir, generationDir, str;
  std::string experimentDir;
  std::vector<ImageClass> imbalancedData;
  Artificial a;
  DIR *dir = NULL;
  std::vector<int> testing_fold, majority_fold, minority_fold, generated_fold;
  std::vector<ImageClass>::iterator itClass, minorityClass;
  std::vector<int>::iterator itOperation;
  std::vector< vector<int> > foldsByGeneration;

  int k = 5;
  if (argc < 3) {
    std::cout << "\nUsage: ./rebalanceTest (0) (1) (2) (3) (4)\n " << std::endl;
    std::cout << "\t(0) Directory to place tests\n" << std::endl;
    std::cout << "\t(1) Image Directory\n" << std::endl;
    exit(-1);
  }
  newDir = std::string(argv[1]);
  baseDir = std::string(argv[2]);

  srand(time(NULL));
  //  Available methods:
  // Description {"BIC", "GCH", "CCV", "Haralick6", "ACC", "LBP", "HOG", "Contour"}
  // Quantization {"Intensity", "Luminance", "Gleam", "MSB", "MSB-Modified", "BGR", "HSV"}

  // std::vector <int> descriptors {0, 5, 6};
  // std::vector <int> quant {0, 2, 1};

  std::vector <int> descriptors {0};
  std::vector <int> quant {0};
  std::vector <int> operations {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  Rebalance r;
  r.extractor.ccvThreshold = 25;
  r.extractor.accDistances = {1, 3, 5, 7};
  r.extractor.normalization = 0;
  r.extractor.resizeFactor = 1.0;
  r.extractor.numColors = 64;

  r.readImageDirectory(baseDir);
  // r.performFeatureExtraction(d, m);
  // Arrange original images in k folds each
  r.separateInFolds(k);
  // r.writeFeatures(newDir+"/original-folds");

  // minorityClass = r.data.smallerTrainingClass();
  experiment = 0;
  for (minorityClass = r.data.classes.begin(); minorityClass != r.data.classes.end(); ++minorityClass) {

    for (i = 0; i < k; i++) {
      for (j = 0; j < k; j++) {
        if (j != i) {
          // Select which folds are going to be training and testing
          for (itClass = r.data.classes.begin(); itClass != r.data.classes.end(); ++itClass) {
            itClass->testing_fold.clear();
            itClass->training_fold.clear();
            itClass->generated_fold.clear();
            itClass->smote_fold.clear();

            itClass->testing_fold.push_back(i);
            if (itClass->id == minorityClass->id) {
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

          experimentDir = newDir+"/artificial/experiment-"+to_string(experiment)+"/";
          featuresDir = experimentDir+"features/";
          analysisDir = experimentDir+"analysis/";
          dir = opendir(experimentDir.c_str());
          if (!dir) {
            // str = "rm -f -r "+generationDir+"/*;";
            str = "mkdir -p "+experimentDir+";";
            str += "mkdir -p "+featuresDir+";";
            str += "mkdir -p "+analysisDir+";";
            system(str.c_str());
            dir = opendir(experimentDir.c_str());
            if (!dir) {
              std::cout << "Error! Directory " << experimentDir;
              std::cout << " can't be created. " << std::endl;
              exit(1);
            }
          }
          closedir(dir);

          std::ofstream FILE(newDir+"/artificial/experiment-"+to_string(experiment)+"/folds.txt", std::ios::out);
          for (itClass = r.data.classes.begin(); itClass != r.data.classes.end(); ++itClass) {
            FILE << "Class " << itClass->id << "\n Testing-folds: ";
            std::copy(itClass->testing_fold.begin(), itClass->testing_fold.end(), std::ostream_iterator<int>(FILE, " "));
            FILE << "\n Training-folds: ";
            std::copy(itClass->training_fold.begin(), itClass->training_fold.end(), std::ostream_iterator<int>(FILE, " "));
            FILE << "\n";
          }
          FILE.close();

          foldsByGeneration.clear();

          for (itOperation = operations.begin(); itOperation != operations.end(); ++itOperation) {
            generationDir = experimentDir+"/operation-"+to_string(*itOperation)+"/";
            dir = opendir(generationDir.c_str());
          	if (!dir) {
          		str = "mkdir -p "+generationDir+";";
          		system(str.c_str());
          		dir = opendir(generationDir.c_str());
          		if (!dir) {
          			std::cout << "Error! Directory " << generationDir;
          			std::cout << " can't be created. " << std::endl;
          			exit(1);
          		}
          	}
          	closedir(dir);

            // Perform images artificial generation
            generated_fold = a.generateImagesFromData(&r.data, generationDir, *itOperation);
            foldsByGeneration.push_back(generated_fold);
          }

          for (indexDescriptor = 0; indexDescriptor < (int)descriptors.size(); indexDescriptor++) {
            d = descriptors[indexDescriptor];
            m = quant[indexDescriptor];

            r.performFeatureExtraction(d, m, false);

            // Perform SMOTE subsampling
            r.performSmote(&r.data, 1);

            r.writeFeatures(featuresDir+"unbalanced_");

            // Classify original folds
            c.classify(0.5,  1, r.data, analysisDir+r.extractor.getName()+"_"+r.quantization.getName()+"_unbalanced_", 1);

            // Add smote fold as training
            for (itClass = r.data.classes.begin(); itClass != r.data.classes.end(); ++itClass) {
              for (x = 0; x < (int) itClass->smote_fold.size(); x++) {
                itClass->training_fold.push_back(itClass->smote_fold[x]);
              }
            }
            // Classify using smote fold as training
            r.writeFeatures(featuresDir+"smote_");
            c.classify(0.5,  1, r.data, analysisDir+r.extractor.getName()+"_"+r.quantization.getName()+"_smote_", 1);
            for (itClass = r.data.classes.begin(); itClass != r.data.classes.end(); ++itClass) {
              for (x = 0; x < (int) itClass->smote_fold.size(); x++) {
                // Remove smote fold as training
                itClass->training_fold.erase(std::remove(itClass->training_fold.begin(),
                                                        itClass->training_fold.end(),
                                                        itClass->smote_fold[x]),
                                            itClass->training_fold.end());
                int current_fold = itClass->smote_fold[x];
                // Remove all smote data
                itClass->images.erase(std::remove_if(itClass->images.begin(),
                                                    itClass->images.end(),
                                                    [current_fold] (Image y) {
                                                      return (y.fold == current_fold);
                                                    }),
                                      itClass->images.end());
              }
            }

            for (operation = 0; operation < (int)foldsByGeneration.size(); operation++) {
              // Add generated fold as training
              for (itClass = r.data.classes.begin(); itClass != r.data.classes.end(); ++itClass) {
                for (x = 0; x < (int) foldsByGeneration[operation].size(); x++) {
                  itClass->training_fold.push_back(foldsByGeneration[operation][x]);
                }
              }
              // Classify using generated fold as training
              r.writeFeatures(featuresDir+"artificial_"+to_string(operations[operation])+"_");
              c.classify(0.5,  1, r.data, analysisDir+r.extractor.getName()+"_"+r.quantization.getName()+"_artificial_"+to_string(operations[operation])+"_", 1);
              for (itClass = r.data.classes.begin(); itClass != r.data.classes.end(); ++itClass) {
                for (x = 0; x < (int) foldsByGeneration[operation].size(); x++) {
                  // Remove generated fold as training
                  itClass->training_fold.erase(std::remove(itClass->training_fold.begin(),
                                                          itClass->training_fold.end(),
                                                          foldsByGeneration[operation][x]),
                                              itClass->training_fold.end());
                }
              }
            }
          }

          for (itClass = r.data.classes.begin(); itClass != r.data.classes.end(); ++itClass) {
            for (x = 0; x < (int) itClass->generated_fold.size(); x++) {
              int current_fold = itClass->generated_fold[x];
              // Remove all generated data
              itClass->images.erase(std::remove_if(itClass->images.begin(),
                                                  itClass->images.end(),
                                                  [current_fold] (Image y) {
                                                    return (y.fold == current_fold);
                                                  }),
                                    itClass->images.end());
            }
            for (x = 0; x < (int) itClass->smote_fold.size(); x++) {
              int current_fold = itClass->smote_fold[x];
              // Remove all generated data
              itClass->images.erase(std::remove_if(itClass->images.begin(),
                                                  itClass->images.end(),
                                                  [current_fold] (Image y) {
                                                    return (y.fold == current_fold);
                                                  }),
                                    itClass->images.end());
            }
          }

          experiment++;
        }
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
