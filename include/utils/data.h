#ifndef _FUNCOES_ARQUIVO_H
#define _FUNCOES_ARQUIVO_H

#include <stdlib.h>
#include <string.h>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <queue>
#include <sys/types.h>
#include <dirent.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include "description/descritores.h"
#include "utils/funcoesAux.h"
#include "quantization/quantization.h"

using namespace cv;
using namespace std;

struct Image {
	Mat features;
	string path;
	int fold;
};

struct ImageClass {
	int id;
	vector<Image> images;
	vector<int> training_fold, testing_fold, smote_fold, generated_fold;
};

class Data {
	public:
		int grayscaleMethod, featureExtractionMethod;
		string originalDirectory, featuresDirectory, analysisDirectory;
		vector<ImageClass> classes;
};

int Data::numClasses(void);
int Data::numTrainingImages(int class);
int Data::numTestingImages(int class);
int Data::biggestClass(void);
int Data::smallerClass(void);
int Data::isFreeTrainOrTest(int class, int fold);
int Data::isOriginalSmoteOrGenerated(int class, int fold);
void Data::addImage(int class, Image img);
bool Data::readFeaturesFromFile(string filename);
string Data::writeFeatures(string id);

#endif
