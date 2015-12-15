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

void Rebalance::ShuffleImages(vector<Image> *img){

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
void Rebalance::SeparateInFolds(vector<ImageClass> *original_data, int k) {

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

vector<vector<double> > Rebalance::classify(string descriptorFile, int repeat, double prob, string csv) {
  Classifier c;
  int minoritySize, numClasses;
  vector<vector<double> > fscore;
  // Read the feature vectors
  vector<ImageClass> data = ReadFeaturesFromFile(descriptorFile);
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

void Rebalance::performFeatureExtraction(vector<ImageClass> *data, int extractMethod, int grayMethod) {

  int dataClass, image;
  Mat img, newimg;

  cout << "\n---------------------------------------------------------" << endl;
  cout << "Image feature extraction using " << extractor.getName(extractMethod-1);
  cout << " and " << quantization.getName(grayMethod-1) << endl;
  cout << "-----------------------------------------------------------" << endl;

  for (dataClass = 0; dataClass < (*data).size(); dataClass++) {
    for (image = 0; image < (*data)[dataClass].images.size(); image++) {

      img = imread((*data)[dataClass].images[image].path, CV_LOAD_IMAGE_COLOR);
      if (!img.empty()) {

        img.copyTo(newimg);
        if (resizeFactor != 1.0) {
          // Resize the image given the input factor
          cv::resize(img, newimg, Size(), resizeFactor, resizeFactor, INTER_AREA);
        }

        // Convert the image to grayscale
        quantization.convert(quantization, newimg, &newimg);

        // Call the description method
        extractor.extract(method, newimg, &(*data)[dataClass].images[image].features);
        if ((*data)[dataClass].images[image].features.cols == 0) {
          cout << "Error: the feature vector is null" << endl;
          exit(1);
        }

        img.release();
        newimg.release();
      }
  }
}

string Rebalance::PerformSmote(vector<ImageClass> imbalancedData, int operation) {

  int biggestTraining, trainingInThisClass, amountSmote, numFeatures;
  int neighbors;
  std::vector<ImageClass>::iterator it;
  Mat synthetic;
  SMOTE s;
  string name;

  biggestTraining = imbalancedData.numTrainingImages(imbalancedData.biggestTrainingClass());

  for (itClass = imbalancedData.begin(); itClass != imbalancedData.end(); ++itClass) {

    /* Find out how many samples are needed to rebalance */
    trainingInThisClass = imbalancedData.numTrainingImages(itClass->id);
    cout << "In class " << itClass->id << " were found " << trainingInThisClass << " original training images"<< endl;
    amountSmote = biggestTraining - trainingInThisClass;
    cout << "Amount to smote: " << amountSmote << " samples" << endl;
    numFeatures = imbalancedData.numFeatures(itClass->id);

    if (amountSmote > 0) {
      Mat dataTraining(0, numFeatures, CV_32FC1);

      //neighbors = 5;
      neighbors = (double)biggestTraining/(double)trainingInThisClass;
      cout << "Number of neighbors: " << neighbors << endl;
      cout << "Number of images in class: " << itClass->images.size() << endl;

      for (itImage = imbalancedData[eachClass].images.begin();
          itImage != imbalancedData[eachClass].images.end();
          ++itImage){
        if (imbalancedData.isTraining(imbalancedData[eachClass].id, itImage->fold)) {
          dataTraining.push_back(itImage->features);
        }
      }

      cout << "Number of training in class: " << dataTraining.size() << endl;
      if (dataTraining.rows > 0) {
        if (operation != 0){
          synthetic = s.smote(dataTraining, amountSmote, neighbors);
        }
        else {
          synthetic.create(amountSmote, numFeatures, CV_32FC1);
          for (x = 0; x < amountSmote; x++){
            pos = rand() % (dataTraining.size().height);
            Mat tmp = synthetic.row(x);
            dataTraining.row(pos).copyTo(tmp);
          }
        }

        cout << "SMOTE generated " << amountSmote << " new synthetic samples" << endl;

        Image newSample;
        /* Concatenate original with synthetic data*/
        for (index = 0; index < synthetic.rows; index++) {
          synthetic.row(index).copyTo(newSample.features);
          newSample.isFreeTrainOrTest = 1;
          newSample.isGenerated = true;
          newSample.path = "smote";
          itClass->images.push_back(newSample);
        }
        total += imgClass.images.size(); //??
        dataTraining.release();
        synthetic.release();
      }
    }
  }

  name = imbalancedData.writeFeatures("smote");

  return name;
}
