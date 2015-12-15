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
