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

#include "utils/rebalance.h"


void Rebalance::writeFeatures(std::string id) {

  std::string name;

  // Decide the features file's name
  name = featuresDirectory;
  name += extractor.getName(extractor.method-1) + "_";
  name += quantization.getName(quantization.method-1) + "_";
  name += std::to_string(extractor.numColors) + "_";
  name += std::to_string(extractor.resizeFactor) + "_";
  name += id + ".csv";

  data.writeFeatures(id, name);
}

void Rebalance::ShuffleImages(std::vector<Image> *img) {

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
void Rebalance::SeparateInFolds(std::vector<ImageClass> *original_data, int k) {

  int i, j, fold, numClasses, numImages, resto, imgsInThisFold, next;
  cv::Mat labels, trainTest, isGenerated, img;

  numClasses = (*original_data).size();
  for (i = 0; i < numClasses; i++) {
    resto = 0; next = 0;

    Rebalance::ShuffleImages(&(*original_data)[i].images);

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

// std::vector<std::vector<double> > Rebalance::classify(std::string descriptorFile, int repeat, double prob, std::string csv) {
//   Classifier c;
//   int minoritySize, numClasses;
//   std::vector<std::vector<double> > fscore;
//   // Read the feature vectors
//   std::vector<ImageClass> data = ReadFeaturesFromFile(descriptorFile);
//   numClasses = data.size();
//   if (numClasses != 0){
//     std::cout << "---------------------------------------------------------------------------------------" << std::endl;
//     std::cout << "Features vectors file: " << descriptorFile.c_str() << std::endl;
//     std::cout << "---------------------------------------------------------------------------------------" << std::endl;
//     c.findSmallerClass(data, &minoritySize);
//     fscore = c.classify(prob, repeat, data, csv.c_str(), minoritySize);
//     data.clear();
//   }
//   return fscore;
// }

void Rebalance::performFeatureExtraction(std::vector<ImageClass> *data, int extractMethod, int grayMethod) {

  int dataClass, image;
  cv::Mat img, newimg;

  std::cout << "\n---------------------------------------------------------" << std::endl;
  std::cout << "Image feature extraction using " << extractor.getName(extractMethod-1);
  std::cout << " and " << quantization.getName(grayMethod-1) << std::endl;
  std::cout << "-----------------------------------------------------------" << std::endl;

  extractor.method = extractMethod;
  quantization.method = grayMethod;

  for (dataClass = 0; dataClass < (int)(*data).size(); dataClass++) {
    for (image = 0; image < (int)(*data)[dataClass].images.size(); image++) {

      img = cv::imread((*data)[dataClass].images[image].path, CV_LOAD_IMAGE_COLOR);
      if (!img.empty()) {

        img.copyTo(newimg);
        if (extractor.resizeFactor != 1.0) {
          // Resize the image given the input factor
          cv::resize(img, newimg, cv::Size(), extractor.resizeFactor, extractor.resizeFactor, CV_INTER_AREA);
        }

        // Convert the image to grayscale
        quantization.convert(grayMethod, newimg, &newimg);

        // Call the description method
        extractor.extract(extractMethod, newimg, &(*data)[dataClass].images[image].features);
        if ((*data)[dataClass].images[image].features.cols == 0) {
          std::cout << "Error: the feature std::vector is null" << std::endl;
          exit(1);
        }

        img.release();
        newimg.release();
      }
    }
  }
}

std::string Rebalance::PerformSmote(Data imbalancedData, int operation) {

  int biggestTraining, trainingInThisClass, amountSmote, numFeatures;
  int neighbors, x, pos, index;
  std::vector<ImageClass>::iterator itClass;
  std::vector<Image>::iterator itImage;
  cv::Mat synthetic;
  SMOTE s;
  std::string name;

  biggestTraining = imbalancedData.numTrainingImages(imbalancedData.biggestTrainingClass());

  for (itClass = imbalancedData.classes.begin();
      itClass != imbalancedData.classes.end();
      ++itClass) {

    /* Find out how many samples are needed to rebalance */
    trainingInThisClass = imbalancedData.numTrainingImages(itClass->id);
    amountSmote = biggestTraining - trainingInThisClass;
    numFeatures = imbalancedData.numFeatures();

    std::cout << "In class " << itClass->id << " were found " << trainingInThisClass;
    std::cout << " original training images"<< std::endl;
    std::cout << "Amount to smote: " << amountSmote << " samples" << std::endl;

    if (amountSmote > 0) {
      cv::Mat dataTraining(0, numFeatures, CV_32FC1);

      //neighbors = 5;
      neighbors = (double)biggestTraining/(double)trainingInThisClass;
      std::cout << "Number of neighbors: " << neighbors << std::endl;
      std::cout << "Number of images in class: " << itClass->images.size() << std::endl;

      for (itImage = itClass->images.begin();
          itImage != itClass->images.end();
          ++itImage){
        if (imbalancedData.isTraining(itClass->id, itImage->fold)) {
          dataTraining.push_back(itImage->features);
        }
      }

      std::cout << "Number of training in class: " << dataTraining.size() << std::endl;
      if (dataTraining.rows > 0) {
        if (operation != 0){
          synthetic = s.smote(dataTraining, amountSmote, neighbors);
        }
        else {
          synthetic.create(amountSmote, numFeatures, CV_32FC1);
          for (x = 0; x < amountSmote; x++){
            pos = rand() % (dataTraining.size().height);
            cv::Mat tmp = synthetic.row(x);
            dataTraining.row(pos).copyTo(tmp);
          }
        }

        std::cout << "SMOTE generated " << amountSmote << " new synthetic samples" << std::endl;

        Image newSample;
        /* Concatenate original with synthetic data*/
        for (index = 0; index < synthetic.rows; index++) {
          synthetic.row(index).copyTo(newSample.features);
          newSample.path = "smote";
          newSample.fold = -1;
          newSample.generationType = -1;
          itClass->images.push_back(newSample);
        }
        dataTraining.release();
        synthetic.release();
      }
    }
  }

  Rebalance::writeFeatures("smote");

  return name;
}

/*******************************************************************************
*******************************************************************************/
int qtdArquivos(std::string directory) {
  int count = 0;
  struct dirent *sDir = NULL;
  DIR *dir = NULL;

  dir = opendir(directory.c_str());
  if (dir == NULL) {
    return 0;
  }

  while ((sDir = readdir(dir))) {
    if ((strcmp(sDir->d_name, ".") != 0) &&
    (strcmp(sDir->d_name, "..") != 0) &&
    (strcmp(sDir->d_name, ".DS_Store") != 0) &&
    (strcmp(sDir->d_name, ".directory") != 0)) {
      count++;
    }
  }

  closedir(dir);
  return count;
}

/*******************************************************************************
*******************************************************************************/
int NumberImagesInDataset(std::string base, int qtdClasses, std::vector<int> *objClass) {
  int i, count = 0, currentSize;
  std::string directory;

  for (i = 0; i < qtdClasses; i++) {
    directory = base + "/" + std::to_string(i) + "/treino/";
    currentSize = qtdArquivos(directory);
    directory = base + "/" + std::to_string(i) + "/treino/generated/";
    currentSize += qtdArquivos(directory);
    directory = base + "/" + std::to_string(i) + "/teste/";
    currentSize += qtdArquivos(directory);
    if (currentSize == 0) {
      directory = base + "/" + std::to_string(i)  + "/";
      currentSize = qtdArquivos(directory);
      if (currentSize == 0) {
        std::cout << "Error: there is no directory named " << directory.c_str();
        exit(-1);
      }
    }
    (*objClass).push_back(currentSize);
    count += currentSize;
  }
  return count;
}

/*******************************************************************************
*******************************************************************************/
cv::Mat FindImgInClass(std::string database, int img_class, int img_number, int index,
                  int treino, cv::Mat *trainTest, std::vector<std::string> *path,
                  cv::Mat *isGenerated) {
  std::string directory, dir_class = database +"/" + std::to_string(img_class);
  cv::Mat img;

  directory = dir_class + "/"+std::to_string(img_number);
  img = cv::imread(directory+".png", CV_LOAD_IMAGE_COLOR);

  // std::cout << directory+".png" << std::endl;
  (*trainTest).at<int>(index, 0) = 0;
  (*isGenerated).at<int>(index, 0) = 0;

  if (img.empty()) {
    directory = dir_class + "/treino/" + std::to_string(img_number);
    img = cv::imread(directory + ".png", CV_LOAD_IMAGE_COLOR);
    (*trainTest).at<int>(index, 0) = 1;

    if (img.empty()) {
      directory = dir_class + "/treino/generated/" + std::to_string(img_number);
      img = cv::imread(directory + ".png", CV_LOAD_IMAGE_COLOR);
      if (img.empty()) {
        directory = dir_class + "/teste/" + std::to_string(img_number - treino);
        img = cv::imread(directory+".png", CV_LOAD_IMAGE_COLOR);
        (*trainTest).at<int>(index, 0) = 2;

        if (img.empty()) {
          std::cout << "Error: there is no image in " << directory.c_str();
          return cv::Mat();
        }
      } else {
        (*isGenerated).at<int>(index, 0) = 1;
      }
    }
  }
  (*path).push_back(directory + ".png");
  return img;
}

/*******************************************************************************
Counts how many images are in a class, and how many of those are for training

Requires
- std::string database name
- int class number
- int* output the total number of images
- int* output the number of images in training
*******************************************************************************/
void NumberImgInClass(std::string database, int img_class, int *num_imgs,
                      int *num_train) {
  std::string directory;

  directory = database + "/" + std::to_string(img_class)  + "/treino/";
  (*num_imgs) = qtdArquivos(directory);
  directory = database + "/" + std::to_string(img_class)  + "/treino/generated/";
  (*num_imgs) += qtdArquivos(directory);
  (*num_train) = (*num_imgs);

  directory = database + "/" + std::to_string(img_class)  + "/teste/";
  (*num_imgs) += qtdArquivos(directory);

  if ((*num_imgs) == 0) {
    directory = database + "/" + std::to_string(img_class)  + "/";
    (*num_imgs) = qtdArquivos(directory);

    if ((*num_imgs) == 0) {
      std::cout << "Error: there is no directory named " << directory.c_str();
      exit(-1);
    }
  }
  std::cout << "class " << img_class << ": " << database + "/" + std::to_string(img_class);
  std::cout << " has " << (*num_imgs) << " images" << std::endl;
}
