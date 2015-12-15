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

#include <vector>

#include "utils/data.h"

int Data::numTrainingImages(int id) {

	int numberImages = 0;
	int fold;
	vector<Image>::iterator itImage;
	vector<ImageClass>::iterator itClass;

	for (itClass = classes.begin(); itClass != classes.end(); ++itClass) {
		if (itClass->id == id){
			for(itImage = itClass->images.begin(); itImage != itClass->images.end(); ++itImage) {
				for (fold = 0; fold < itClass->training_fold.size(); fold++) {
					if (itImage->fold == itClass->training_fold[fold]) {
						numberImages++;
					}
				}
			}
		}
	}

	return numberImages;
}

int Data::numTestingImages(int id) {

	int numberImages = 0;
	int fold;
	vector<Image>::iterator itImage;
	vector<ImageClass>::iterator itClass;

	for (itClass = classes.begin(); itClass != classes.end(); ++itClass) {
		if (itClass->id == id){
			for(itImage = itClass->images.begin(); itImage != itClass->images.end(); ++itImage) {
				for (fold = 0; fold < itClass->testing_fold.size(); fold++) {
					if (itImage->fold == itClass->testing_fold[fold]) {
						numberImages++;
					}
				}
			}
		}
	}

	return numberImages;
}

int Data::biggestClass(void) {

	double biggest = -1;
	int biggestClass = -1;
	vector<ImageClass>::iterator itClass;

	for (itClass = classes.begin(); itClass != classes.end(); ++itClass) {
		trainingImages = numTrainingImages(itClass->id);
		testingImages = numTestingImages(itClass->id);
		if (trainingImages + testingImages > biggest) {
			biggest = trainingImages + testingImages;
			biggestClass = itClass->id;
		}
	}

	return biggestClass;
}

int Data::biggestTrainingClass(void) {

	double biggest = -1;
	int biggestClass = -1;
	vector<ImageClass>::iterator itClass;

	for (itClass = classes.begin(); itClass != classes.end(); ++itClass) {
		trainingImages = numTrainingImages(itClass->id);
		if (trainingImages + testingImages > biggest) {
			biggest = trainingImages + testingImages;
			biggestClass = itClass->id;
		}
	}

	return biggestClass;
}

int Data::smallerClass(void) {

	double smaller = std::numeric_limits<double>::infinity();
	int smallerClass = -1;
	vector<ImageClass>::iterator itClass;

	for (itClass = classes.begin(); itClass != classes.end(); ++itClass) {
		trainingImages = numTrainingImages(itClass->id);
		testingImages = numTestingImages(itClass->id);
		if (trainingImages + testingImages < smaller) {
			smaller = trainingImages + testingImages;
			smallerClass = itClass->id;
		}
	}

	return smallerClass;
}

int Data::isFreeTrainOrTest(int id, int fold) {

	int freeTrainOrTest = 0;
	vector<ImageClass>::iterator itClass;

	for (itClass = classes.begin(); itClass != classes.end(); ++itClass) {
		if (itClass->id == id){
			for (fold = 0; fold < itClass->training_fold.size(); fold++) {
				if (fold == itClass->training_fold[fold]) {
					freeTrainOrTest == 1;
				}
			}
			for (fold = 0; fold < itClass->testing_fold.size(); fold++) {
				if (fold == itClass->testing_fold[fold]) {
					freeTrainOrTest == 2;
				}
			}
		}
	}

	return freeTrainOrTest;
}

bool Data::isTraining(int id, int fold) {
  bool = training;
  training = if Data::isFreeTrainOrTest(id, fold) == 1 ? true : false;
  return training;
}

bool Data::isTesting(int id, int fold) {
  bool = training;
  training = if Data::isFreeTrainOrTest(id, fold) == 2 ? true : false;
  return training;
}


int Data::numClasses(void) {
  return static<int>(classes.size());
}


int Data::numFeatures(void) {

  int num = 0;
  vector<ImageClass>::iterator itClass;
  vector<Images>::iterator itImage;

  for (itClass = classes.begin(); itClass != classes.end(); ++itClass) {
    for (itImage = itClass->images.begin();
        itImage != itClass->images.end();
        ++itImage) {
      num = itImage->features.size();
      return num;
    }
  }

  return num;
}

int Data::isOriginalSmoteOrGenerated(int class, int fold) {

	int freeTrainOrTest = 0;
	vector<ImageClass>::iterator itClass;

	for (itClass = classes.begin(); itClass != classes.end(); ++itClass) {
		if (itClass->id == class){
			for (fold = 0; fold < itClass->smote_fold.size(); fold++) {
				if (fold == itClass->training_fold[fold]) {
					freeTrainOrTest == 1;
				}
			}
			for (fold = 0; fold < itClass->generated_fold.size(); fold++) {
				if (fold == itClass->testing_fold[fold]) {
					freeTrainOrTest == 2;
				}
			}
		}
	}

	return freeTrainOrTest;
}

void Data::addImage(int class, Image img){

	bool ok = false;
	vector<ImageClass>::iterator itClass;

	for (itClass = classes.begin(); itClass != classes.end(); ++itClass) {
		if (itClass->id == class){
			itClass->images.push_back(img);
			ok = true;
		}
	}

	if (!ok) {
		ImageClass newClass;
		newClass.id = class;
		newClass.push_back(img);
		classes.push_back(newClass);
	}
}

/*******************************************************************************
Read the features from the input file and save them in a vector of classes

Requires
- string name of input file
*******************************************************************************/
bool Data::readFeaturesFromFile(string filename) {
  int class;
  string features;
  string line, pathImage, classe, trainTest, numClasses, generated;
  ifstream myFile;
  Image img;

  myFile.open(filename.c_str(), ios::in);
  if (!myFile.is_open()) {
    cout << "It is not possible to open the feature's file: " << filename << endl;
    return false;
  }

  while (getline(myFile, line)) {
    stringstream vector_features(line);
    getline(vector_features, pathImage, ',');
    getline(vector_features, classe, ',');
    getline(vector_features, trainTest, ',');
    getline(vector_features, generated, ',');

    class = atoi(classe.c_str());

  	img.path = pathImage;
  	img.isGenerated = atoi(generated.c_str());
  	img.isFreeTrainOrTest = atoi(trainTest.c_str());
		img.isOriginalSmoteOrGenerated = 0;

		while (!getline(vector_features, features, ',').eof()) {
			img.features.push_back(stof(features));
    }
		getline(vector_features, features, '\n');
    img.features.push_back(stof(features));

		addImage(class, img);
		img.features.clear();
  }

  myFile.close();
}

/*******************************************************************************
Write the features of Mat features in a csv file

Requires
- string directory path to the new csv file
*******************************************************************************/
string Data::writeFeatures(string id) {

  ofstream arq, arqVis;
  int i, j;
  string name;

  // Decide the features file's name
  name = featuresDir;
	name += getFeatureExtractionName(extractionMethod-1) + "_";
  name += getQuantizationName(quantizationMethod-1) + "_";
	name += to_string(colors) + "_";
  name += to_string(resizingFactor) + "_";
  name += to_string(features.rows) + "_";
	name += id + ".csv";

  // Open file to write features
  arq.open(name.c_str(), ios::out);
  if (!arq.is_open()) {
    cout << "It is not possible to open the feature's file: " << name << endl;
    return "";
  }

	// For each class
	for (itClass = classes.begin(); itClass != classes.end(); ++itClass) {
		// For each image
		for(itImage = itClass->images.begin(); itImage != itClass->images.end(); ++itImage) {
			// Write the image name, the respective class and fold
      arq << itImage->path << ',' << itClass->id << ',' << itImage->fold << ',';
			arq << isFreeTrainOrTest(itClass->id, itImage->fold) << ',';
			arq << isOriginalSmoteOrGenerated(itClass->id, itImage->fold) << ',';

			// Write the feature vector related to the current image
			for (j = 0; j < itImage->features.cols-1; j++) {
	      arq << itImage->features.at<float>(i, j) << ",";
	    }
	    arq << itImage->features.at<float>(i, j) << endl;
		}
	}

  cout << "---------------------------------------------------------------------------------------" << endl;
  cout << "Wrote on data file named " << name << endl;
  cout << "---------------------------------------------------------------------------------------" << endl;
  arq.close();
  return name;
}
