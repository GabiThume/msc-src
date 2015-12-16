#ifndef _DATA_H
#define __DATA_H

#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>

struct Image {
	cv::Mat features;
	std::string path;
	int fold;
	int generationType;
};

struct ImageClass {
	int id;
	std::vector<Image> images;
	std::vector<int> training_fold, testing_fold, smote_fold, generated_fold;
};

class Data {
	public:
		int grayscaleMethod, featureExtractionMethod;
		std::vector<ImageClass> classes;

		int numClasses(void);
		int numTrainingImages(int id);
		int numTestingImages(int id);
		int biggestClass(void);
		int biggestTrainingClass(void);
		int smallerClass(void);
		int smallerTrainingClass(void);
		int isFreeTrainOrTest(int id, int fold);
		int isOriginalSmoteOrGenerated(int id, int fold);
		bool readFeaturesFromFile(std::string filename);
		bool isTraining(int id, int fold);
		bool isTesting(int id, int fold);
		void addClass(int id);
		void addImage(int id, Image img);
		void addFoldSmote(int id, int smote_fold);
		void addFoldGenerated(int id, int smote_fold);
		void addFoldTraining(int id, int smote_fold);
		void addFoldTesting(int id, int smote_fold);
		bool writeFeatures(std::string id, std::string name);
		int numFeatures(void);
};

#endif
