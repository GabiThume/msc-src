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
Read the features from the input file and save them in a vector of classes

Requires
- string name of input file
*******************************************************************************/
vector<Classes> ReadFeaturesFromFile(string filename) {
  int j, newSize = 0, previousClass = -1, actualClass;
  float features;
  size_t d;
  string line, infos, numImage, classe, trainTest, numFeatures, numClasses;
  string objetos;
  vector<Classes> data;
  Classes imgClass;
  ifstream myFile;

  myFile.open(filename.c_str(), ios::in);
  if (!myFile.is_open()) {
    cout << "It is not possible to open the feature's file: " << filename << endl;
    exit(-1);
  }

  /* Read the first line, which contains the number of objects,
  classes and features */
  getline(myFile, infos);
  if (infos == "") return Mat();
  stringstream info(infos);
  getline(info, objetos, '\t');
  getline(info, numClasses, '\t');
  getline(info, numFeatures, '\t');

  d = atoi(numFeatures.c_str());

  while (getline(myFile, line)) {
    stringstream vector_features(line);
    getline(vector_features, numImage, '\t');
    getline(vector_features, classe, '\t');
    getline(vector_features, trainTest, '\t');
    actualClass = atoi(classe.c_str());
    if (previousClass != actualClass) {
      if (previousClass != -1) {
        data.push_back(imgClass);
      }
      previousClass = actualClass;
      imgClass.features.create(0, d, CV_32FC1);
      imgClass.trainOrTest.create(0, 1, CV_32S);
      imgClass.fixedTrainOrTest = false;
    }

    newSize = imgClass.features.size().height+1;
    imgClass.features.resize(newSize);
    imgClass.trainOrTest.resize(newSize);

    j = 0;
    while (vector_features >> features) {
      imgClass.features.at<float>(newSize-1, j) = static_cast<float>(features);
      j++;
    }
    imgClass.trainOrTest.at<int>(newSize-1, 0) = atoi(trainTest.c_str());
    imgClass.classNumber = actualClass;
    if (atoi(trainTest.c_str()) != 0) {
      imgClass.fixedTrainOrTest = true;
    }
  }
  if (previousClass != -1) {
    data.push_back(imgClass);
  }

  myFile.close();
  return data;
}

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
                  int treino, Mat *trainTest) {
  string directory, dir_class = database +"/" + to_string(img_class);
  Mat img;

  directory = dir_class + "/"+to_string(img_number);
  img = imread(directory+".png", CV_LOAD_IMAGE_COLOR);

  // cout << directory+".png" << endl;
  (*trainTest).at<int>(index, 0) = 0;

  if (img.empty()) {
    directory = dir_class + "/treino/" + to_string(img_number);
    img = imread(directory + ".png", CV_LOAD_IMAGE_COLOR);
    (*trainTest).at<int>(index, 0) = 1;

    if (img.empty()) {
      directory = dir_class + "/teste/" + to_string(img_number - treino);
      img = imread(directory+".png", CV_LOAD_IMAGE_COLOR);
      (*trainTest).at<int>(index, 0) = 2;

      if (img.empty()) {
        cout << "Error: there is no image in " << directory.c_str();
        exit(-1);
      }
    }
  }
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
Write the features of Mat features in a csv file

Requires
- string directory where to write the csv file
- int quantization method
- int description method
- int number of colors wanted in the image
- int indicate if normalization is necessary (0-None 1-[0,1] 255-[0,255])
- int factor for resizing
- int number of classes
- Mat of features, one row for each image
- Mat of labels, one row for each image
- Mat indicating if the image in each row is fixed for training or testing
- string id to add in the end of file (requested for static rebalance)
- bool indicates if it is necessary to write a data file
*******************************************************************************/
string WriteFeaturesOnFile(string featuresDir, int quantization, int method,
                    int colors, int normalization, int resizingFactor,
                    int qtdClasses, Mat features, Mat labels, Mat trainTest,
                    string id, bool writeDataFile) {

  ofstream arq, arqVis;
  int i, j;
  string name;

  // Decide the features file's name
  name = featuresDir + descriptorMethod[method-1] + "_";
  name += quantizationMethod[quantization-1] + "_" + to_string(colors);
  name += "c_" + to_string(resizingFactor) + "r_";
  name += to_string(features.rows) + "i_" + id + ".csv";

  // Open file to write features
  arq.open(name.c_str(), ios::out);
  if (!arq.is_open()) {
    cout << "It is not possible to open the feature's file: " << name << endl;
    exit(-1);
  }

  // Write the calculated features on a file to load in classification
  // Number of images \t number of classes \t number of features per image
  arq << features.rows << '\t' << qtdClasses << '\t' << features.cols << endl;
  // For each image
  for (i = 0; i < features.rows; i++) {
    if (labels.rows > 0) {
      // Write the image number \t the referenced class \t
      arq << i << '\t' << labels.at<int>(i, 0) << '\t';
      // and if it is fixed as training or testing image
      arq << trainTest.at<int>(i, 0) << '\t';
    }
    // Write the feature vector related to the current image
    for (j = 0; j < features.cols; j++) {
      arq << features.at<float>(i, j) << " ";
    }
    arq << endl;
  }
  arq.close();

  // Write a DATA file if requested (expected in some visualizations tools)
  if (writeDataFile) {
    arqVis.open((name+"data").c_str(), ios::in);
    arqVis << "DY\n" << labels.size().height << '\n';
    arqVis << features.size().width << '\n';
    for (i = 0; i < features.size().width-1; i++) {
      arqVis << "attr" << i << ";";
    }
    arqVis << "attr" << i << "\n";
    for (i = 0; i < labels.size().height; i++) {
      arqVis << i << ".png";
      for (j = 0; j < features.size().width; j++) {
        arqVis << features.at<float>(i, j) << ";";
      }
      int numeroimg = labels.at<int>(i, 0);
      arqVis << numeroimg << endl;
    }
    arqVis.close();
  }

  return name;
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
      (*features).at<float>(row, column) =
        ((*features).at<float>(row, column) - (float) mean.val[0])
        / (float) std.val[0];
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
  Mat img, featureVector, features, labels, trainTest, newimg;
  vector<int> num_images_class;
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

    for (i = 0; i < qtdClasses; i++) {
      NumberImgInClass(database, i, &numImages, &treino);

      for (j = 0; j < numImages; j++)    {
        // The image label is the i index
        labels.at<int>(imgTotal, 0) = i;
        // Find this image in the class and open it
        img = FindImgInClass(database, i, j, imgTotal, treino, &trainTest);
        imgTotal++;

        // Resize the image given the input factor
        img.copyTo(newimg);
        if (resizeFactor != 1) {
          resize(img, newimg, Size(), resizeFactor, resizeFactor, INTER_AREA);
        }

        // Convert the image to grayscale
        ConvertToGrayscale(quantization, newimg, &newimg, colors);

        // Call the description method
        GetFeatureVector(method, newimg, &featureVector, colors, normalization,
          param);

        // Push the feature vector for the current image in the features vector
        features.push_back(featureVector);
        featureVector.release();
        img.release();
        newimg.release();
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
    id, false);
  cout << "File: " << name << endl;
  end = clock();
  cout << endl << "Elapsed time: " << double(end-begin)/ CLOCKS_PER_SEC << endl;

  return name;
}
