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

#include <string>
#include <vector>
#include "utils/funcoesArquivo.h"

/* Read the features and save them in Mat data */
vector<Classes> readFeatures(string filename) {
  int j, newSize = 0;
  float features;
  size_t d;
  string line, infos, numImage, classe, trainTest, numFeatures, numClasses;
  string objetos;
  vector<Classes> data;
  int previousClass = -1, actualClass;
  Classes imgClass;

  ifstream myFile(filename.c_str());
  if (!myFile) throw exception();

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
      imgClass.trainOrTest.create(0, 1, CV_32FC1);
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
    imgClass.trainOrTest.at<float>(newSize-1, 0) = atoi(trainTest.c_str());
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

int qtdImagensTotal(string base, int qtdClasses, vector<int> *objClass,
                    int *maxs) {
  int i, count = 0, currentSize;
  string directory;
  *maxs = 0;

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
    if (currentSize > *maxs || *maxs == 0) {
      *maxs = currentSize;
    }
  }

  return count;
}

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
    default:
      cout << "Error: quantization method " << method;
      cout << " does not exists." << endl;
      exit(1);
  }
  namedWindow("Display window", WINDOW_AUTOSIZE );
  imshow("Grayscale Image", *gray);
  waitKey(0);
}

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

Mat FindImgInClass(string database, int img_class, int img_number, int index,
                  int treino, Mat *trainTest) {
  string directory, dir_class = database +"/" + to_string(img_class);
  Mat img;

  directory = dir_class + "/"+to_string(img_number);
  img = imread(directory+".png", CV_LOAD_IMAGE_COLOR);
  (*trainTest).at<uchar>(index, 0) = static_cast<uchar>(0);

  if (img.empty()) {
    directory = dir_class + "/treino/" + to_string(img_number);
    img = imread(directory + ".png", CV_LOAD_IMAGE_COLOR);
    (*trainTest).at<uchar>(index, 0) = static_cast<uchar>(1);

    if (img.empty()) {
      directory = dir_class + "/teste/" + to_string(img_number - treino);
      img = imread(directory+".png", CV_LOAD_IMAGE_COLOR);
      (*trainTest).at<uchar>(index, 0) = static_cast<uchar>(2);

      if (img.empty()) {
        cout << "Error: there is no image in " << directory.c_str();
        exit(-1);
      }
    }
  }

  return img;
}

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

void ZIndexNormalization(Mat *features, int normalization) {
  float normFactor, min, max, value;
  int i, j;

  normFactor = (normalization == 1) ? 1.0 : 255.0;

  for (j = 0; j < (*features).cols; ++j) {
    min = (*features).at<float>(0, j);
    max = (*features).at<float>(0, j);

    for (i = 1; i < (*features).rows; ++i) {
      value = (*features).at<float>(i, j);
      if (value > max) {
        max = value;
      }
      if (value < min) {
        min = value;
      }
    }

    for (i = 0; i < (*features).rows; ++i) {
      (*features).at<float>(i, j) = normFactor *
        (((*features).at<float>(i, j) - min) / (max - min));
    }
  }
}

string WriteFeaturesOnFile(string featuresDir, int quantization, int method,
                    int colors, int normalization, int resizingFactor,
                    int qtdClasses, vector<int>objperClass, Mat features, Mat labels,
                    Mat trainTest, string id, bool writeDataFile) {

  FILE *arq;
  int i, j, bars;
  float porc;
  string nome;

  nome = featuresDir + descriptorMethod[method-1] + "_";
  nome += quantizationMethod[quantization-1] + "_" + to_string(colors);
  nome += "c_" + to_string(resizingFactor) + "r_";
  nome += to_string(features.rows) + "i_" + id + ".csv";
  arq = fopen(nome.c_str(), "w+");
  if (arq == 0) {
    cout << "It is not possible to open the feature's file: " << nome << endl;
    exit(-1);
  }

  fprintf(arq, "%d\t%d\t%d\n", features.rows, qtdClasses, features.cols);
  cout << "File: " << nome << endl;
  cout << "Number of images: " << features.rows << " - Classes: " << qtdClasses;
  cout << " - Features: " << features.cols << endl;

  for (i = 0; i < qtdClasses; i++) {
    bars = (static_cast<float> (objperClass[i]) /
      static_cast<double> (features.rows)) * 50.0;
    cout << i << " ";
    for (j = 0; j < bars; j++) {
      cout << "|";
    }
    porc = static_cast<double> (objperClass[i]) /
      static_cast<double> (features.rows);
    cout << " " << porc * 100 << "%" << " (" << objperClass[i] << ")" <<endl;
  }

  for (i = 0; i < features.rows; i++) {
    // Write the image number and the referenced class
    if (labels.rows > 0) {
      fprintf(arq, "%d\t%d\t%d\t", i, labels.at<uchar>(i, 0),
        trainTest.at<uchar>(i, 0));
    }

    for (j = 0; j < features.cols; j++) {
      if (normalization == 2)  {
        fprintf(arq, "%.f ", features.at<float>(i, j));
      } else {
        fprintf(arq, "%.5f ", features.at<float>(i, j));
      }
    }
    fprintf(arq, "\n");
  }
  fclose(arq);

  // Write a DATA file if requested
  if (writeDataFile) {
    cout << "Wrote on data file named " << nome << endl;
    FILE *arqVis = fopen((nome+"data").c_str(), "w+");
    fprintf(arqVis, "%s\n", "DY");
    fprintf(arqVis, "%d\n", labels.size().height);
    fprintf(arqVis, "%d\n", features.size().width);
    for (i = 0; i < features.size().width-1; i++) {
      fprintf(arqVis, "%s%d;", "attr", i);
    }
    fprintf(arqVis, "%s%d\n", "attr", i);
    for (i = 0; i < labels.size().height; i++) {
      fprintf(arqVis, "%d%s;", i, ".png");
      for (j = 0; j < features.size().width; j++) {
        fprintf(arqVis, "%.5f;", features.at<float>(i, j));
      }
      float numeroimg = labels.at<uchar>(i, 0);
      fprintf(arqVis, "%1.1f\n", numeroimg);
    }
    fclose(arqVis);
  }
  return nome;
}

string descriptor(string database, string featuresDir, int method, int colors,
                  double resizeFactor, int normalization, vector<int> param,
                  int deleteNull, int quantization, string id = "") {
  int numImages = 0, qtdClasses = 0, qtdImgTotal = 0, imgTotal = 0, treino = 0;
  int maxc = 0, i, j;
  int resizingFactor = static_cast<int>(resizeFactor*100);
  string nome, directory;
  Mat img, featureVector, features, labels, trainTest, newimg;
  vector<int> objperClass;

  cout << "\n---------------------------------------------------------" << endl;
  cout << "Image feature extraction using " << descriptorMethod[method-1];
  cout << " and " << quantizationMethod[quantization-1] << endl;
  cout << "-----------------------------------------------------------" << endl;

  cout << "Database: " << database << endl;

  img = imread(database, CV_LOAD_IMAGE_COLOR);
  if (!img.empty()) {
    resize(img, newimg, Size(), resizeFactor, resizeFactor, INTER_AREA);
    ConvertToGrayscale(quantization, newimg, &newimg, colors);
    GetFeatureVector(method, newimg, &features, colors, normalization, param);

    featureVector.release();
    img.release();
    newimg.release();
  } else {
    // Check how many classes and images there are
    qtdClasses = qtdArquivos(database+"/");
    qtdImgTotal = qtdImagensTotal(database, qtdClasses, &objperClass, &maxc);
    labels = Mat::zeros(qtdImgTotal, 1, CV_8U);
    trainTest = Mat::zeros(qtdImgTotal, 1, CV_8U);

    for (i = 0; i < qtdClasses; i++) {
      NumberImgInClass(database, i, &numImages, &treino);

      for (j = 0; j < numImages; j++)    {
        labels.at<uchar>(imgTotal, 0) = (uchar)i;
        img = FindImgInClass(database, i, j, imgTotal, treino, &trainTest);
        imgTotal++;

        resize(img, newimg, Size(), resizeFactor, resizeFactor, INTER_AREA);

        ConvertToGrayscale(quantization, newimg, &newimg, colors);
        GetFeatureVector(method, newimg, &featureVector, colors, normalization, param);
        features.push_back(featureVector);
        // exit(1);
        featureVector.release();
        img.release();
        newimg.release();
      }
    }
    // Normalization of Haralick and contourExtraction features by z-index
    if ((method == 4 || method == 8) && normalization != 0) {
      ZIndexNormalization(&features, normalization);
    }
  }

  nome = WriteFeaturesOnFile(featuresDir, quantization, method, colors,
    normalization, resizingFactor, qtdClasses, objperClass, features, labels,
    trainTest, id, false);

  return nome;
}
