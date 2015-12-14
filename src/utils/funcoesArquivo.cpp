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

#include <ctime>
#include <string>
#include <vector>
#include "utils/funcoesArquivo.h"

/*******************************************************************************
*******************************************************************************/
int qtdArquivos(string directory) {
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
int NumberImagesInDataset(string base, int qtdClasses, vector<int> *objClass) {
  int i, count = 0, currentSize;
  string directory;

  for (i = 0; i < qtdClasses; i++) {
    directory = base + "/" + to_string(i) + "/treino/";
    currentSize = qtdArquivos(directory);
    directory = base + "/" + to_string(i) + "/treino/generated/";
    currentSize += qtdArquivos(directory);
    directory = base + "/" + to_string(i) + "/teste/";
    currentSize += qtdArquivos(directory);
    if (currentSize == 0) {
      directory = base + "/" + to_string(i)  + "/";
      currentSize = qtdArquivos(directory);
      if (currentSize == 0) {
        cout << "Error: there is no directory named " << directory.c_str();
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
Mat FindImgInClass(string database, int img_class, int img_number, int index,
                  int treino, Mat *trainTest, vector<string> *path,
                  Mat *isGenerated) {
  string directory, dir_class = database +"/" + to_string(img_class);
  Mat img;

  directory = dir_class + "/"+to_string(img_number);
  img = imread(directory+".png", CV_LOAD_IMAGE_COLOR);

  // cout << directory+".png" << endl;
  (*trainTest).at<int>(index, 0) = 0;
  (*isGenerated).at<int>(index, 0) = 0;

  if (img.empty()) {
    directory = dir_class + "/treino/" + to_string(img_number);
    img = imread(directory + ".png", CV_LOAD_IMAGE_COLOR);
    (*trainTest).at<int>(index, 0) = 1;

    if (img.empty()) {
      directory = dir_class + "/treino/generated/" + to_string(img_number);
      img = imread(directory + ".png", CV_LOAD_IMAGE_COLOR);
      if (img.empty()) {
        directory = dir_class + "/teste/" + to_string(img_number - treino);
        img = imread(directory+".png", CV_LOAD_IMAGE_COLOR);
        (*trainTest).at<int>(index, 0) = 2;

        if (img.empty()) {
          cout << "Error: there is no image in " << directory.c_str();
          return Mat();
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
- string database name
- int class number
- int* output the total number of images
- int* output the number of images in training
*******************************************************************************/
void NumberImgInClass(string database, int img_class, int *num_imgs,
                      int *num_train) {
  string directory;

  directory = database + "/" + to_string(img_class)  + "/treino/";
  (*num_imgs) = qtdArquivos(directory);
  directory = database + "/" + to_string(img_class)  + "/treino/generated/";
  (*num_imgs) += qtdArquivos(directory);
  (*num_train) = (*num_imgs);

  directory = database + "/" + to_string(img_class)  + "/teste/";
  (*num_imgs) += qtdArquivos(directory);

  if ((*num_imgs) == 0) {
    directory = database + "/" + to_string(img_class)  + "/";
    (*num_imgs) = qtdArquivos(directory);

    if ((*num_imgs) == 0) {
      cout << "Error: there is no directory named " << directory.c_str();
      exit(-1);
    }
  }
  cout << "class " << img_class << ": " << database + "/" + to_string(img_class);
  cout << " has " << (*num_imgs) << " images" << endl;
}

/*******************************************************************************
    Remove null columns in feature space (Under construction)

    Requires:
    - Mat features
*******************************************************************************/
void RemoveNullColumns(Mat *features) {
  vector<float> features_col_summed;
  int i;

  cv::reduce(*features, features_col_summed, 0, CV_REDUCE_SUM);
  for (i = 0; i < (*features).cols; i++) {
    if (features_col_summed[i] == 0) {
      // cout << " column " << i << " (*features).cols " << (*features).cols << endl;
      if (i == 0) {
        (*features).colRange(i+1, (*features).cols).copyTo(*features);
      } else if (i+1 == (*features).cols) {
        (*features).colRange(0, i).copyTo(*features);
      } else {
        Mat start = (*features).colRange(0, i);
        Mat end = (*features).colRange(i+1, (*features).cols);
        hconcat(start, end, *features);
      }
      i--;
    }
  }
}

/*******************************************************************************
Normalization by Z-score of each column

1 - Calculate the column mean
2 - Calculate the column std
3 - Calculate new values using:
      new_value = (value - mean) / standart_deviation

*******************************************************************************/
void ZScoreNormalization(Mat *features) {
  int row, column;
  Scalar mean, std;

  for (column = 0; column < (*features).cols; ++column) {
    Mat stats = (*features).col(column);
    cv::meanStdDev(stats, mean, std);

    for (row = 0; row < (*features).rows; ++row) {
      cout << "(*features).at<float>(row, column) " << (*features).at<float>(row, column) << endl;
      (*features).at<float>(row, column) =
        ((*features).at<float>(row, column) - (float) mean.val[0])
        / (float) std.val[0];
      cout << "(float) mean.val[0]) " << (float) mean.val[0] << endl;
      cout << "(float) std.val[0] " << (float) std.val[0] << endl;
      cout << "result " << (*features).at<float>(row, column) << endl << endl;
    }
  }
}

void MaxMinNormalization(Mat *features, int normalization) {
  float normFactor, min, max, value;
  int row, column;

  normFactor = (normalization == 1) ? 1.0 : 255.0;

  for (column = 0; column < (*features).cols; ++column) {
    Mat current_column = (*features).col(column);

    min = (*features).at<float>(0, column);
    max = (*features).at<float>(0, column);

    // Find max and min value in cols
    for (row = 1; row < (*features).rows; ++row) {
      value = (*features).at<float>(row, column);
      if (value > max) {
        max = value;
      }
      if (value < min) {
        min = value;
      }
    }

    for (row = 0; row < (*features).rows; ++row) {
      (*features).at<float>(row, column) = normFactor *
        (((*features).at<float>(row, column) - min) / (max - min));
    }
  }
}

/*******************************************************************************
*******************************************************************************/
void ConvertToGrayscale(int method, Mat img, Mat *gray, int colors) {
  switch (method) {
    case 1:
      QuantizationIntensity(img, gray, colors);
      break;
    case 2:
      QuantizationLuminance(img, gray, colors);
      break;
    case 3:
      QuantizationGleam(img, gray, colors);
      break;
    case 4:
      QuantizationMSB(img, gray, colors);
      break;
    case 5:
      QuantizationMSBModified(img, gray, colors);
      break;
    case 6: // Keep in BRG
      break;
    case 7: // Convert BGR -> HSV
      cvtColor(img, (*gray), CV_BGR2HSV);
      break;
    default:
      cout << "Error: quantization method " << method;
      cout << " does not exists." << endl;
      exit(1);
  }
  // namedWindow("Display window", WINDOW_AUTOSIZE );
  // imshow("Grayscale Image", *gray);
  // waitKey(0);
}

/*******************************************************************************
*******************************************************************************/
void GetFeatureVector(int method, Mat img, Mat *featureVector, int colors,
                      int normalization, vector<int> param) {

  switch (method) {
    case 1:
      BIC(img, featureVector, colors, normalization);
      break;
    case 2:
      GCH(img, featureVector, colors, normalization);
      break;
    case 3:
      CCV(img, featureVector, colors, normalization, param[0]);
      break;
    case 4:
      HARALICK(img, featureVector, colors, normalization);
      break;
    case 5:
      ACC(img, featureVector, colors, normalization, param);
      break;
    case 6:
      LBP(img, featureVector, colors, normalization);
      break;
    case 7:
      HOG(img, featureVector, colors, normalization);
      break;
    case 8:
      contourExtraction(img, featureVector, colors, normalization);
      break;
    default:
      cout << "Error: this description method " << method;
      cout << " does not exists." << endl;
      exit(1);
  }
  if ((*featureVector).cols == 0) {
    cout << "Error: the feature vector is null" << endl;
    exit(-1);
  }
}

/*******************************************************************************
*******************************************************************************/
string PerformFeatureExtraction(string database, string featuresDir, int method,
    int colors, double resizeFactor, int normalization, vector<int> param,
    int deleteNull, int quantization, string id){
  int numImages = 0, qtdClasses = 0, qtdImgTotal = 0, imgTotal = 0, treino = 0;
  int i, j, bars, current_class;
  double porc;
  int resizingFactor = static_cast<int>(resizeFactor*100);
  string name, directory;
  Mat img, featureVector, features, labels, trainTest, newimg, isGenerated;
  vector<int> num_images_class;
  vector<string> path;
  clock_t begin, end;

  cout << "\n---------------------------------------------------------" << endl;
  cout << "Image feature extraction using " << descriptorMethod[method-1];
  cout << " and " << quantizationMethod[quantization-1] << endl;
  cout << "-----------------------------------------------------------" << endl;

  cout << "Database: " << database << endl;

  begin = clock();
  img = imread(database, CV_LOAD_IMAGE_COLOR);
  if (!img.empty()) {
    // Resize the image given the input factor
    resize(img, newimg, Size(), resizeFactor, resizeFactor, INTER_AREA);

    // Convert the image to grayscale
    ConvertToGrayscale(quantization, newimg, &newimg, colors);

    // Call the description method
    GetFeatureVector(method, newimg, &features, colors, normalization, param);

    img.release();
    newimg.release();
  } else {
    // Check how many classes and images there are
    qtdClasses = qtdArquivos(database+"/");
    qtdImgTotal = NumberImagesInDataset(database, qtdClasses, &num_images_class);
    labels = Mat::zeros(qtdImgTotal, 1, CV_32S);
    trainTest = Mat::zeros(qtdImgTotal, 1, CV_32S);
    isGenerated= Mat::zeros(qtdImgTotal, 1, CV_32S);


    for (i = 0; i < qtdClasses; i++) {
      NumberImgInClass(database, i, &numImages, &treino);

      for (j = 0; j < numImages; j++)    {
        // The image label is the i index
        labels.at<int>(imgTotal, 0) = i;
        // Find this image in the class and open it
        img = FindImgInClass(database, i, j, imgTotal, treino, &trainTest, &path,
                            &isGenerated);
        if (!img.empty()) {

          // Resize the image given the input size
          img.copyTo(newimg);
          if (resizeFactor != 1.0) {
            cv::resize(img, newimg, Size(), resizeFactor, resizeFactor, INTER_AREA);
          }

          // Convert the image to grayscale
          ConvertToGrayscale(quantization, newimg, &newimg, colors);

          // Call the description method
          GetFeatureVector(method, newimg, &featureVector, colors, normalization,
            param);

          // Push the feature vector for the current image in the features vector
          features.push_back(featureVector);
          imgTotal++;
          featureVector.release();
          img.release();
          newimg.release();
        }
      }
    }
    // Normalization of Haralick and contourExtraction features by z-index
    if ((method == 4 || method == 8) && normalization) {
      ZScoreNormalization(&features);
    }

    // Remove null columns in Mat of features
    if (deleteNull) {
      RemoveNullColumns(&features);
    }
  }

  // Show the number of images per class
  cout << "Images: " << features.rows << " - Classes: " << qtdClasses;
  cout << " - Features: " << features.cols << endl;
  for (current_class = 0; current_class < qtdClasses; current_class++) {
    bars = (static_cast<double> (num_images_class[current_class]) /
      static_cast<double> (features.rows)) * 50.0;
    cout << current_class << " ";
    for (j = 0; j < bars; j++) {
      cout << "|";
    }
    porc = static_cast<double> (num_images_class[current_class]) /
      static_cast<double> (features.rows);
    cout << " " << porc * 100.0 << "%" << " (";
    cout << num_images_class[current_class] << ")" <<endl;
  }

  name = WriteFeaturesOnFile(featuresDir, quantization, method, colors,
    normalization, resizingFactor, qtdClasses, features, labels, trainTest,
    isGenerated, path, id, false);
  cout << "File: " << name << endl;
  end = clock();
  cout << endl << "Elapsed time: " << double(end-begin)/ CLOCKS_PER_SEC << endl;

  return name;
}


void PerformFeatureExtractionFromData(vector<Classes> *data, int method,
    int colors, double resizeFactor, int normalization, vector<int> param,
    int deleteNull, int quantization){

  int dataClass, image;
  int resizingFactor = static_cast<int>(resizeFactor*100);
  Mat img, newimg;

  cout << "\n---------------------------------------------------------" << endl;
  cout << "Image feature extraction using " << descriptorMethod[method-1];
  cout << " and " << quantizationMethod[quantization-1] << endl;
  cout << "-----------------------------------------------------------" << endl;

  for (dataClass = 0; dataClass < (*data).size(); dataClass++) {
    for (image = 0; image < (*data)[dataClass].images.size(); image++) {

      img = imread((*data)[dataClass].images[image].path, CV_LOAD_IMAGE_COLOR);
      if (!img.empty() and !(*data)[dataClass].images[image].features.empty()) {

        img.copyTo(newimg);
        if (resizeFactor != 1.0) {
          // Resize the image given the input factor
          cv::resize(img, newimg, Size(), resizeFactor, resizeFactor, INTER_AREA);
        }

        // Convert the image to grayscale
        ConvertToGrayscale(quantization, newimg, &newimg, colors);

        // Call the description method
        GetFeatureVector(method, newimg, &(*data)[dataClass].images[image].features, colors, normalization, param);

        img.release();
        newimg.release();
      }
  }
}
