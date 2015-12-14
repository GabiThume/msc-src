int Data::numClasses(void) {
	return static<int>(classes.size());
}

int Data::numTrainingImages(int class) {

	int numberImages = 0;
	int fold;
	vector<Image>::iterator itImage;
	vector<ImageClass>::iterator itClass;

	for (itClass = classes.begin(); itClass != classes.end(); ++itClass) {
		if (itClass->id == class){
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

int Data::numTestingImages(int class) {

	int numberImages = 0;
	int fold;
	vector<Image>::iterator itImage;
	vector<ImageClass>::iterator itClass;

	for (itClass = classes.begin(); itClass != classes.end(); ++itClass) {
		if (itClass->id == class){
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

int Data::isFreeTrainOrTest(int class, int fold) {

	int freeTrainOrTest = 0;
	vector<ImageClass>::iterator itClass;

	for (itClass = classes.begin(); itClass != classes.end(); ++itClass) {
		if (itClass->id == class){
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

  arq.close();
  return name;
}
