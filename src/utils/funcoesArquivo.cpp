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

string descriptor(string database, string featuresDir, int method, int colors,
                  double resizeFactor, int normalization, vector<int> param,
                  int deleteNull, int quantization, string id = "") {
  int numImages = 0, qtdClasses = 0, qtdImgTotal = 0, imgTotal = 0, treino = 0;
  int maxc = 0, i, j, k;
  int resizingFactor = static_cast<int>(resizeFactor*100);
  float min, max, normFactor, value;
  string nome, directory;
  Mat img, featureVector, features, labels, trainTest, newimg;
  FILE *arq;
  vector<int> objperClass;

  cout << "\n---------------------------------------------------------" << endl;
  cout << "Image feature extraction using " << descriptorMethod[method-1];
  cout << " and " << quantizationMethod[quantization-1] << endl;
  cout << "-----------------------------------------------------------" << endl;

  // Check how many classes and images there are
  directory = database+"/";
  qtdClasses = qtdArquivos(directory);
  qtdImgTotal = qtdImagensTotal(database, qtdClasses, &objperClass, &maxc);

  labels = Mat::zeros(qtdImgTotal, 1, CV_8U);
  trainTest = Mat::zeros(qtdImgTotal, 1, CV_8U);

  for (i = 0; i < qtdClasses; i++) {
    directory = database + "/" + to_string(i)  + "/treino/";
    numImages = qtdArquivos(directory);
    treino = numImages;

    directory = database + "/" + to_string(i)  + "/teste/";
    numImages += qtdArquivos(directory);

    if (numImages == 0) {
      directory = database + "/" + to_string(i)  + "/";
      numImages = qtdArquivos(directory);
      if (numImages == 0) {
        cout << "Error: there is no directory named " << directory.c_str();
        exit(-1);
      }
    }

    cout << "class " << i << ": " << database + "/" + to_string(i);
    cout << " has " << numImages << " images" << endl;

    for (j = 0; j < numImages; j++)    {
      directory = database +"/"+to_string(i)+"/"+to_string(j);
      img = imread(directory+".jpg", CV_LOAD_IMAGE_COLOR);
      if (img.empty()) {
        img = imread(directory+".png", CV_LOAD_IMAGE_COLOR);
      }
      trainTest.at<uchar>(imgTotal, 0) = static_cast<uchar>(0);
      if (img.empty()) {
        directory = database + "/" + to_string(i) + "/treino/" + to_string(j);
        img = imread(directory + ".jpg", CV_LOAD_IMAGE_COLOR);
        if (img.empty()) {
          img = imread(directory + ".png", CV_LOAD_IMAGE_COLOR);
        }
        trainTest.at<uchar>(imgTotal, 0) = static_cast<uchar>(1);
        if (img.empty()) {
          directory = database + "/" + to_string(i) + "/teste/";
          directory += to_string(j-treino);
          img = imread(directory + ".jpg", CV_LOAD_IMAGE_COLOR);
          if (img.empty()) {
            img = imread(directory+".png", CV_LOAD_IMAGE_COLOR);
          }
          trainTest.at<uchar>(imgTotal, 0) = static_cast<uchar>(2);
          if (img.empty()) {
            cout << "Error: there is no image in " << directory.c_str();
            exit(-1);
          }
        }
      }

      if (resizeFactor < 1) {
        resize(img, newimg, Size(), resizeFactor, resizeFactor, INTER_AREA);
      } else {
        img.copyTo(newimg);
      }

      switch (quantization) {
        case 1:
          QuantizationIntensity(newimg, &newimg, colors);
          break;
        case 2:
          QuantizationLuminance(newimg, &newimg, colors);
          break;
        case 3:
          QuantizationGleam(newimg, &newimg, colors);
          break;
        case 4:
          QuantizationMSB(newimg, &newimg, colors);
          break;
        default:
          cout << "Error: quantization method does not exists." << endl;
          exit(1);
      }

      switch (method) {
        case 1:
          BIC(newimg, &featureVector, colors, normalization);
          break;
        case 2:
          GCH(newimg, &featureVector, colors, normalization);
          break;
        case 3:
          CCV(newimg, &featureVector, colors, normalization, param[0]);
          break;
        case 4:
          HARALICK(newimg, &featureVector, colors, normalization);
          break;
        case 5:
          ACC(newimg, &featureVector, colors, normalization, param);
          break;
        case 6:
          LBP(newimg, &featureVector, colors, normalization);
          break;
        case 7:
          HOG(newimg, &featureVector, colors, normalization);
          break;
        case 8:
          contourExtraction(newimg, &featureVector, colors, normalization);
          break;
        default:
          cout << "Error: this description method does not exists." << endl;
          exit(1);
      }

      if (featureVector.cols == 0) {
        cout << "Error: the feature vector is null" << endl;
        exit(-1);
      }

      labels.at<uchar>(imgTotal, 0) = (uchar)i;
      if (features.size().height == 0) {
        features = Mat::zeros(0, featureVector.size().width, CV_32F);
      }
      if (features.cols != featureVector.cols) {
        cout << "Erro: descriptor generated different vectors sizes " << endl;
        exit(-1);
      }
      imgTotal++;
      features.push_back(featureVector);
      featureVector.release();
      img.release();
      newimg.release();
    }
  }

  if ((method == 4 || method == 8) && normalization != 0) {
    normFactor = (normalization == 1) ? 1.0 : 255.0;

    for (j = 0; j < features.cols; ++j) {
      min = features.at<float>(0, j);
      max = features.at<float>(0, j);

      for (i = 1; i < features.rows; ++i) {
        value = features.at<float>(i, j);
        if (value > max) {
          max = value;
        }
        if (value < min) {
          min = value;
        }
      }

      for (i = 0; i < features.rows; ++i) {
        features.at<float>(i, j) = normFactor *
          ((features.at<float>(i, j) - min) / (max - min));
      }
    }
  }

  nome = featuresDir+descriptorMethod[method-1] + "_";
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
  cout << "Objects: " << features.rows << " - Classes: " << qtdClasses;
  cout << " - Features: " << features.cols << endl;
  for (i = 0; i < qtdClasses; i++) {
    int bars = (static_cast<float> (objperClass[i]) /
      static_cast<double> (features.rows)) * 50.0;
    cout << i << " ";
    for (j = 0; j < bars; j++) {
      cout << "|";
    }
    float porc = static_cast<double> (objperClass[i]) /
      static_cast<double> (features.rows);
    cout << " " << porc * 100 << "%" << " (" << objperClass[i] << ")" <<endl;
  }

  for (i = 0; i < features.rows; i++) {
    // Write the image number and the referenced class
    fprintf(arq, "%d\t%d\t%d\t", i, labels.at<uchar>(i, 0),
      trainTest.at<uchar>(i, 0));
    for (k = 0; k < features.cols; k++) {
      if (normalization == 2)  {
        fprintf(arq, "%.f ", features.at<float>(i, k));
      } else {
        fprintf(arq, "%.5f ", features.at<float>(i, k));
      }
    }
    fprintf(arq, "\n");
  }
  fclose(arq);

  bool writeDataFile = false;
  if (writeDataFile) {
    cout << "Wrote on data file named " << nome << endl;
    FILE *arqVis = fopen((nome+"data").c_str(), "w+");
    int w, z;
    fprintf(arqVis, "%s\n", "DY");
    fprintf(arqVis, "%d\n", labels.size().height);
    fprintf(arqVis, "%d\n", features.size().width);
    for (z = 0; z < features.size().width-1; z++) {
      fprintf(arqVis, "%s%d;", "attr", z);
    }
    fprintf(arqVis, "%s%d\n", "attr", z);
    for (w = 0; w < labels.size().height; w++) {
      fprintf(arqVis, "%d%s;", w, ".png");
      for (z = 0; z < features.size().width; z++) {
        fprintf(arqVis, "%.5f;", features.at<float>(w, z));
      }
      float numeroimg = labels.at<uchar>(w, 0);
      fprintf(arqVis, "%1.1f\n", numeroimg);
    }
  }

  return nome;
}
