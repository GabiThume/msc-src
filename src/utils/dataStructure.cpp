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

#include "utils/dataStructure.h"

void Data::release(void) {
	std::vector<Image>::iterator itImage;
	std::vector<ImageClass>::iterator itClass;

	for (itClass = classes.begin(); itClass != classes.end(); ++itClass) {
		itClass->training_fold.clear();
		itClass->testing_fold.clear();
		itClass->smote_fold.clear();
		itClass->generated_fold.clear();
		for(itImage = itClass->images.begin();
				itImage != itClass->images.end();
				++itImage) {
			itImage->features.release();
		}
		itClass->images.clear();
	}
	classes.clear();
}

int Data::newFold(int id) {
	int fold = 0;
	std::vector<ImageClass>::iterator itClass;
	std::vector<int>::iterator itFold;

	for (itClass = classes.begin(); itClass != classes.end(); ++itClass) {
		if (itClass->id == id){
			while(fold < (int) itClass->images.size()) {
				if (std::find(itClass->training_fold.begin(),
											itClass->training_fold.end(),
											fold)
						!= itClass->training_fold.end()) {
					fold++;
				}
				else if (std::find(itClass->testing_fold.begin(),
													itClass->testing_fold.end(),
													fold)
								!= itClass->testing_fold.end()) {
					fold++;
				}
				else if (std::find(itClass->original_fold.begin(),
													itClass->original_fold.end(),
													fold)
								!= itClass->original_fold.end()) {
					fold++;
				}
				else if (std::find(itClass->smote_fold.begin(),
													itClass->smote_fold.end(),
													fold)
								!= itClass->smote_fold.end()) {
					fold++;
				}
				else if (std::find(itClass->generated_fold.begin(),
													itClass->generated_fold.end(),
													fold)
								!= itClass->generated_fold.end()) {
					fold++;
				}
				else {
					return fold;
				}
			}
		}
	}

	return -1;
}

int Data::numTrainingImages(int id) {
	int numberImages = 0;
	std::vector<Image>::iterator itImage;
	std::vector<ImageClass>::iterator itClass;

	for (itClass = classes.begin(); itClass != classes.end(); ++itClass) {
		if (itClass->id == id){
			for(itImage = itClass->images.begin(); itImage != itClass->images.end(); ++itImage) {
        if (Data::isTraining(itClass->id, itImage->fold)) {
					numberImages++;
				}
			}
		}
	}
	return numberImages;
}

int Data::numTestingImages(int id) {
	int numberImages = 0;
	std::vector<Image>::iterator itImage;
	std::vector<ImageClass>::iterator itClass;

	for (itClass = classes.begin(); itClass != classes.end(); ++itClass) {
		if (itClass->id == id){
			for(itImage = itClass->images.begin(); itImage != itClass->images.end(); ++itImage) {
				if (Data::isTesting(itClass->id, itImage->fold)) {
					numberImages++;
				}
			}
		}
	}
	return numberImages;
}

int Data::biggestClass(void) {
	double biggest = -1;
	int biggestClass = -1, trainingImages, testingImages;
	std::vector<ImageClass>::iterator itClass;

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
	int biggestClass = -1, trainingImages;
	std::vector<ImageClass>::iterator itClass;

	for (itClass = classes.begin(); itClass != classes.end(); ++itClass) {
		trainingImages = numTrainingImages(itClass->id);
		if (trainingImages > biggest) {
			biggest = trainingImages;
			biggestClass = itClass->id;
		}
	}
	return biggestClass;
}

int Data::biggestTrainingNumber(void) {
	int biggest = -1, trainingImages = 0;
	std::vector<ImageClass>::iterator itClass;

	for (itClass = classes.begin(); itClass != classes.end(); ++itClass) {
		trainingImages = numTrainingImages(itClass->id);
		if (trainingImages > biggest) {
			biggest = trainingImages;
		}
	}
	return biggest;
}

int Data::biggestNumber(void) {
	int biggest = -1, numImages = 0;
	std::vector<ImageClass>::iterator itClass;

	for (itClass = classes.begin(); itClass != classes.end(); ++itClass) {
		numImages = itClass->images.size();
		if (numImages > biggest) {
			biggest = numImages;
		}
	}
	return biggest;
}

int Data::smallerClass(void) {
	double smaller = std::numeric_limits<double>::infinity();
	int smallerClass = -1, trainingImages, testingImages;
	std::vector<ImageClass>::iterator itClass;

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

int Data::smallerTrainingClass(void) {
	double smaller = std::numeric_limits<double>::infinity();
	int smallerClass = -1, trainingImages;
	std::vector<ImageClass>::iterator itClass;

	for (itClass = classes.begin(); itClass != classes.end(); ++itClass) {
		trainingImages = numTrainingImages(itClass->id);
		if (trainingImages < smaller) {
			smaller = trainingImages;
			smallerClass = itClass->id;
		}
	}
	return smallerClass;
}

int Data::isFreeTrainOrTest(int id, int fold) {
  std::vector<ImageClass>::iterator itClass;
	for (itClass = classes.begin(); itClass != classes.end(); ++itClass) {
		if (itClass->id == id){
      if (std::find(itClass->training_fold.begin(),
                    itClass->training_fold.end(),
                    fold)
        != itClass->training_fold.end()) {
            return 1;
      }
      if (std::find(itClass->testing_fold.begin(),
                    itClass->testing_fold.end(),
                    fold)
        != itClass->testing_fold.end()) {
            return 2;
      }
		}
	}
	return 0;
}

int Data::isOriginalSmoteOrGenerated(int id, int fold) {
	std::vector<ImageClass>::iterator itClass;
	for (itClass = classes.begin(); itClass != classes.end(); ++itClass) {
		if (itClass->id == id){
      if (std::find(itClass->smote_fold.begin(),
                    itClass->smote_fold.end(),
                    fold)
        != itClass->smote_fold.end()) {
            return 1;
      }
      if (std::find(itClass->generated_fold.begin(),
                    itClass->generated_fold.end(),
                    fold)
        != itClass->generated_fold.end()) {
            return 2;
      }
		}
	}
	return 0;
}

bool Data::isTraining(int id, int fold) {
  return (Data::isFreeTrainOrTest(id, fold) == 1) ? true : false;
}

bool Data::isTesting(int id, int fold) {
  return (Data::isFreeTrainOrTest(id, fold) == 2) ? true : false;
}

int Data::numClasses(void) {
  return classes.size();
}

int Data::numImages(void) {
  int num = 0;
  std::vector<ImageClass>::iterator itClass;
  for (itClass = classes.begin(); itClass != classes.end(); ++itClass) {
    num += itClass->images.size();
  }
  return num;
}

int Data::isBalanced(void) {
  int num = -1;
  std::vector<ImageClass>::iterator itClass;
  for (itClass = classes.begin(); itClass != classes.end(); ++itClass) {
    if (num == -1) {
      num = itClass->images.size();
    }
    else {
      if (num != (int) itClass->images.size()) {
        return false;
      }
    }
  }
  return true;
}


int Data::numFeatures(void) {
  int num = 0;
  std::vector<ImageClass>::iterator itClass;
  std::vector<Image>::iterator itImage;

  for (itClass = classes.begin(); itClass != classes.end(); ++itClass) {
    for (itImage = itClass->images.begin();
        itImage != itClass->images.end();
        ++itImage) {
      num = itImage->features.cols;
      return num;
    }
  }
  return num;
}

void Data::addClass(int id) {
  bool exist = false;

  std::vector<ImageClass>::iterator itClass;
  for (itClass = classes.begin(); itClass != classes.end(); ++itClass) {
		if (itClass->id == id){
      exist = true;
    }
  }

  if (!exist) {
    ImageClass newClass;
    newClass.id = id;
    classes.push_back(newClass);
  }
}

void Data::addImage(int id, Image img) {
	std::vector<ImageClass>::iterator itClass;
  addClass(id);
	for (itClass = classes.begin(); itClass != classes.end(); ++itClass) {
		if (itClass->id == id){
			itClass->images.push_back(img);
		}
	}
}

void Data::addFoldSmote(int id, int smote_fold) {
	std::vector<ImageClass>::iterator itClass;
	for (itClass = classes.begin(); itClass != classes.end(); ++itClass) {
		if (itClass->id == id){
      if (std::find(itClass->smote_fold.begin(),
                    itClass->smote_fold.end(),
                    smote_fold)
          == itClass->smote_fold.end()) {
        itClass->smote_fold.push_back(smote_fold);
      }
		}
	}
}

void Data::addFoldGenerated(int id, int generated_fold) {
	std::vector<ImageClass>::iterator itClass;
	for (itClass = classes.begin(); itClass != classes.end(); ++itClass) {
		if (itClass->id == id){
      if (std::find(itClass->generated_fold.begin(),
                    itClass->generated_fold.end(),
                    generated_fold)
          == itClass->generated_fold.end()) {
        itClass->generated_fold.push_back(generated_fold);
      }
		}
	}
}

void Data::addFoldTraining(int id, int training_fold) {
	std::vector<ImageClass>::iterator itClass;
	for (itClass = classes.begin(); itClass != classes.end(); ++itClass) {
		if (itClass->id == id){
      if (std::find(itClass->training_fold.begin(),
                    itClass->training_fold.end(),
                    training_fold)
          == itClass->training_fold.end()) {
        itClass->training_fold.push_back(training_fold);
      }
		}
	}
}

void Data::addFoldTesting(int id, int testing_fold) {
	std::vector<ImageClass>::iterator itClass;
	for (itClass = classes.begin(); itClass != classes.end(); ++itClass) {
		if (itClass->id == id){
      if (std::find(itClass->testing_fold.begin(),
                    itClass->testing_fold.end(),
                    testing_fold)
          == itClass->testing_fold.end()) {
        itClass->testing_fold.push_back(testing_fold);
      }
		}
	}
}

/*******************************************************************************
Read the features from the input file and save them in a std::vector of classes

Requires
- std::string name of input file
*******************************************************************************/
bool Data::readFeaturesFromFile(std::string filename) {
  int id;
  std::string features, fold;
  std::string line, pathImage, classe, isFreeTrainOrTest, numClasses, generationType;
  std::ifstream myFile;
  Image img;

  myFile.open(filename.c_str(), std::ios::in);
  if (!myFile.is_open()) {
    std::cout << "It is not possible to open the feature's file: " << filename << std::endl;
    return false;
  }

  while (getline(myFile, line)) {
    std::stringstream vector_features(line);
    getline(vector_features, pathImage, ',');
    getline(vector_features, classe, ',');
    getline(vector_features, fold, ',');
    getline(vector_features, isFreeTrainOrTest, ',');
    getline(vector_features, generationType, ',');

    id = atoi(classe.c_str());
    addClass(id);

    img.path = pathImage;
    img.fold = atoi(fold.c_str());
    if (img.path.compare("smote")) {
      addFoldSmote(id, img.fold);
    }
    if (atoi(isFreeTrainOrTest.c_str()) == 1) {
      addFoldTraining(id, img.fold);
    }
    else if (atoi(isFreeTrainOrTest.c_str()) == 2) {
      addFoldTesting(id, img.fold);
    }
  	img.generationType = atoi(generationType.c_str());
    if (img.generationType != -1) {
      addFoldGenerated(id, img.fold);
    }

		while (!getline(vector_features, features, ',').eof()) {
			img.features.push_back(stof(features));
    }
		getline(vector_features, features, '\n');
    img.features.push_back(stof(features));

		addImage(id, img);
		img.features.release();
  }

  myFile.close();
  return true;
}

/*******************************************************************************
Write the features of cv::Mat features in a csv file
	path, id, fold, isFreeTrainOrTest, isOriginalSmoteOrGenerated, features

Requires
- std::string name of the new csv file
*******************************************************************************/
bool Data::writeFeatures(std::string name) {

  std::ofstream arq;
  std::vector<ImageClass>::iterator itClass;
  std::vector<Image>::iterator itImage;
  int col;

  // Open file to write features
  arq.open(name.c_str(), std::ios::out);
  if (!arq.is_open()) {
    std::cout << "It is not possible to open the feature's file: " << name << std::endl;
    return false;
  }

	// For each class
	for (itClass = classes.begin(); itClass != classes.end(); ++itClass) {
		// For each image
		for(itImage = itClass->images.begin(); itImage != itClass->images.end(); ++itImage) {
			if (itImage->features.cols) {
				// Write the image name, the respective class and fold
	      arq << itImage->path << ',' << itClass->id << ',' << itImage->fold << ',';
				arq << isFreeTrainOrTest(itClass->id, itImage->fold) << ',';
				arq << isOriginalSmoteOrGenerated(itClass->id, itImage->fold) << ',';

				// Write the feature std::vector related to the current image
				for (col = 0; col < itImage->features.cols-1; col++) {
		      arq << itImage->features.at<float>(0, col) << ",";
		    }
		    arq << itImage->features.at<float>(0, col) << std::endl;
			}
		}
	}

  std::cout << "Wrote current features on data file named " << name << std::endl;
  arq.close();
  return true;
}
