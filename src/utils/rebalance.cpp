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
#include <sys/stat.h>
#include "utils/rebalance.h"

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

void Rebalance::readImageDirectory(std::string directory) {

  cv::Mat img;
  std::string dir, filepath, classPath;
  DIR *root, *newImageClass;
  struct dirent *classesDir, *imagesInClass;
  struct stat filestat;
  int classId = 0;
  Image newImage;

  std::cout << "Images Directory: " << directory << std::endl;

  // If it is a path to an image
  img = cv::imread(directory, CV_LOAD_IMAGE_COLOR);
  if (!img.empty()) {
    img.release();
  } else { // If it is a directory
    root = opendir(directory.c_str());
    if (root == NULL) {
      std::cout << "Error while trying to open " << directory << std::endl;
    } else {
      imagesDirectory = directory;
      // For each file inside this directory
      while ((classesDir = readdir(root))) {
        if ((strcmp(classesDir->d_name, ".") != 0) &&
            (strcmp(classesDir->d_name, "..") != 0) &&
            (strcmp(classesDir->d_name, ".DS_Store") != 0) &&
            (strcmp(classesDir->d_name, ".directory") != 0)) {

          classPath = imagesDirectory + "/" + classesDir->d_name;
          // Is is invalid, skip it
          if (stat(classPath.c_str(), &filestat)) continue;
          // If it isn't, try to open and add it as an class
          if (S_ISDIR(filestat.st_mode)) {
            newImageClass = opendir(classPath.c_str());
            if (newImageClass == NULL) {
              std::cout << "Error while trying to open " << classPath << std::endl;
            } else {
              std::cout << "New class in " << classPath << std::endl;
              data.addClass(classId);
              // For each file inside it, try to read as an image
              while ((imagesInClass = readdir(newImageClass))) {
                filepath = classPath + "/" + imagesInClass->d_name;
                // Is is invalid, skip it
                if (stat(filepath.c_str(), &filestat)) continue;
                img = cv::imread(filepath, CV_LOAD_IMAGE_COLOR);
                if (!img.empty()) {
                  std::cout << "Read " << filepath << std::endl;
                  newImage.path = filepath;
                  newImage.fold = -1;
                  newImage.generationType = -1;
                  data.addImage(classId, newImage);
                  img.release();
                }
              }
              classId++;
            }
          }
        } else {
          // Try to read to check if it is an image
          img = cv::imread(imagesDirectory, CV_LOAD_IMAGE_COLOR);
          if (!img.empty()) {
            img.release();
          }
        }
      }
    }
  }
  std::cout << "Reading done." << std::endl;
}

void Rebalance::writeFeatures(std::string id) {

  std::string name;

  std::cout << "-------------------------------------------------" << std::endl;
  // Decide the features file's name
  name = featuresDirectory;
  name += extractor.getName(extractor.method-1) + "_";
  name += quantization.getName(quantization.method-1) + "_";
  name += std::to_string(extractor.numColors) + "_";
  name += std::to_string(data.numFeatures()) + "_";
  name += id + ".csv";

  data.writeFeatures(name);
}

void Rebalance::shuffleImages(std::vector<Image> *img) {

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
void Rebalance::separateInFolds(int k) {

  int i, j, fold, numClasses, numImages, resto, imgsInThisFold, next;
  cv::Mat labels, trainTest, isGenerated, img;

  std::cout << "-------------------------------------------------" << std::endl;
  std::cout << "Arrange images in " << k << " folds each " << std::endl;

  numClasses = data.classes.size();
  for (i = 0; i < numClasses; i++) {
    resto = 0; next = 0;

    Rebalance::shuffleImages(&data.classes[i].images);

    numImages = data.classes[i].images.size();
    for (fold = 0; fold < k; fold++) {
      // If the number of images per fold is less than the total, increase each extra image in a fold
      imgsInThisFold = floor(numImages/k);
      if (numImages % k > resto) {
        imgsInThisFold++;
        resto++;
      }
      // For each image requested for this fold, add the fold indication
      for (j = 0; j < imgsInThisFold; j++) {
        data.classes[i].images[next].fold = fold;
        next++;
      }
      if (imgsInThisFold > 0) {
        data.classes[i].original_fold.push_back(fold);
      }
    }
  }
  std::cout << "Separation in folds done." << std::endl;
}

void Rebalance::performFeatureExtraction(int extractMethod, int grayMethod) {

  int dataClass, image;
  cv::Mat img, newimg;

  std::cout << "-------------------------------------------------" << std::endl;
  std::cout << "Image feature extraction using " << extractor.getName(extractMethod-1);
  std::cout << " and " << quantization.getName(grayMethod-1) << std::endl;

  extractor.method = extractMethod;
  quantization.method = grayMethod;

  for (dataClass = 0; dataClass < (int)data.classes.size(); dataClass++) {
    for (image = 0; image < (int)data.classes[dataClass].images.size(); image++) {

      img = cv::imread(data.classes[dataClass].images[image].path, CV_LOAD_IMAGE_COLOR);
      if (!img.empty()) {
        img.copyTo(newimg);
        if (extractor.resizeFactor != 1.0) {
          // Resize the image given the input factor
          cv::resize(img, newimg, cv::Size(), extractor.resizeFactor, extractor.resizeFactor, CV_INTER_AREA);
        }

        // Convert the image to grayscale
        quantization.convert(64, grayMethod, newimg, &newimg);

        // Call the description method
        extractor.extract(64, 1, extractMethod, newimg, &data.classes[dataClass].images[image].features);
        if (data.classes[dataClass].images[image].features.cols == 0) {
          std::cout << "Error: the feature vector is empty" << std::endl;
          exit(1);
        }

        img.release();
        newimg.release();
      }
    }
  }
  std::cout << "Feature extraction done." << std::endl;
}

std::string Rebalance::performSmote(Data imbalancedData, int operation) {

  int biggestTraining, trainingInThisClass, amountSmote, numFeatures;
  int neighbors, x, pos, index;
  std::vector<ImageClass>::iterator itClass;
  std::vector<Image>::iterator itImage;
  cv::Mat synthetic;
  SMOTE s;
  std::string name;

  std::cout << "-------------------------------------------------" << std::endl;
  std::cout << "SMOTE " << std::endl;

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

  std::cout << "SMOTE done." << std::endl;
  return name;
}
