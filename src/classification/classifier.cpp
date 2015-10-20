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

#include "classification/classifier.h"

/* Train and predict using the Normal Bayes classifier */
void Classifier::bayesClassifier(Mat dataTraining, Mat labelsTraining,
		Mat dataTesting, Mat& result){
	CvNormalBayesClassifier classifier;
	classifier.train(dataTraining, labelsTraining);
	classifier.predict(dataTesting, &result);
	classifier.clear();
}

/* Train and predict using 1-KNN */
void Classifier::knn(Mat dataTraining, Mat labelsTraining, Mat dataTesting,
		int k, Mat& result) {
	Mat responses, dist, nearests(1, k, CV_32F);
	CvKNearest knn(dataTraining, labelsTraining, responses, false, k);
	knn.find_nearest(dataTesting, k, result, nearests, dist);
	knn.clear();
	nearests.release();
	responses.release();
	dist.release();
}

double Classifier::calculateMean(vector<double> accuracy) {
	int i;
	double mean;
	/* Calculate accuracy's mean */
	mean = 0.0;
	for (i = 0; i < (int) accuracy.size(); i++){
		mean += accuracy[i];
	}
	mean = mean/accuracy.size();

	if (std::isnan(mean))
		mean = 0.0;

	return mean;
}

double Classifier::calculateStandardDeviation(vector<double> accuracy) {
	int i;
	double mean, variance, std;
	mean = calculateMean(accuracy);
	/* Calculate accuracy's variance and std */
	variance = 0.0;
	for (i = 0; i < (int) accuracy.size(); i++){
		variance += pow(accuracy[i]-mean, 2);
	}
	variance = variance/accuracy.size();
	std = sqrt(variance);

	if (std::isnan(std))
		std = 0.0;

	return std;
}

void Classifier::printAccuracy(double id, vector<vector<double> > fScore) {
	int i;
	ofstream outputFile;
	vector<double> fscores, fscoresStd;
	double mean, std, balancedMean, balancedStd, fscoreMean, fscoreStd;
	mean = calculateMean(accuracy);
	std = calculateStandardDeviation(accuracy);
	balancedMean = calculateMean(balancedAccuracy);
	balancedStd = calculateStandardDeviation(balancedAccuracy);

	for (i = 0; i < (int) fScore.size(); i++){
		fscores.push_back(calculateMean(fScore[i]));
	}
	fscoreMean = calculateMean(fscores);

	for (i = 0; i < (int) fScore.size(); i++){
		fscoresStd.push_back(calculateStandardDeviation(fScore[i]));
	}
	fscoreStd = calculateStandardDeviation(fscoresStd);

	cout << "-----------------------------------------------------------" << endl;
	cout << "Image classification using KNN classifier" << endl;
	cout << "-----------------------------------------------------------" << endl;
	cout << "Number of classes: " << numClasses << endl;
	cout << "Total samples: " << totalTest + totalTrain << endl;
	cout << "\tTesting samples: " << totalTest << endl;
	cout << "\tTraining samples: " << totalTrain << endl;
	cout <<  "Cross validation with "<< accuracy.size() <<" k-fold:" << endl;
	cout << "\tMean Accuracy: " << mean << endl;
	if (std != 0)
		cout << "\t\tStandard Deviation: " << std << endl;
	cout << "\tMean Balanced Accuracy: " << balancedMean << endl;
	if (balancedStd != 0)
		cout << "\t\tStandard Deviation: " << balancedStd << endl;
	cout << "\tMean F1Score: " << fscoreMean << endl;
	if (fscoreStd != 0)
		cout << "\t\tStandard Deviation: " << fscoreStd << endl;
	cout << "-----------------------------------------------------------" << endl;

	if (outputName != ""){
		cout << "Write on " << (outputName+"FScore.csv").c_str() << endl;
		cout << "---------------------------------------------------------" << endl;

		// outputFile.open((outputName+"BalancedAccuracy.csv").c_str(), ios::out | ios::app);
		// outputFile << id << "," << balancedMean << "\n";
		// outputFile.close();

		outputFile.open((outputName+"FScore.csv").c_str(), ios::out | ios::app);
		outputFile << id << "," << fscoreMean << "\n";
		// outputFile << fscoreMean << "\n";
		// for (i = 0; i < (int) fscores.size(); i++){
		// 	outputFile << i << "," << fscores[i] << "\n";
		// }
		outputFile.close();
	}
}

/* Find which is the smaller class */
int Classifier::findSmallerClass(vector<Classes> imageClasses,
		int *minoritySize) {
	int class_id = 0, minorityClass;
	std::vector<Classes>::iterator it;
	(*minoritySize) = imageClasses[0].features.size().height;

	for(it = imageClasses.begin(); it != imageClasses.end(); ++it) {
		if (it->features.size().height < (*minoritySize)){
			(*minoritySize) = it->features.size().height;
			minorityClass = class_id;
		}
		class_id++;
	}
	return minorityClass;
}

double calculateBalancedAccuracy(Mat confusionMat) {
	double balancedAccuracyMean, truePositive, falseNegative, falsePositive;
	double trueNegative, specificity, sensitivity, positive, negative;
	int class_id, i, j;

	balancedAccuracyMean = 0.0;
	for (class_id = 0; class_id < confusionMat.rows; class_id++){

		truePositive = falseNegative = falsePositive = trueNegative = 0;
		truePositive = confusionMat.at<int>(class_id, class_id);

		for (i = 0; i < confusionMat.rows; i++) {
			for (j = 0; j < confusionMat.cols; j++) {
				if (i == class_id && j != class_id) {
					falseNegative += (double) confusionMat.at<int>(i, j);
				}
				if (i != class_id && j == class_id) {
					falsePositive += (double) confusionMat.at<int>(i, j);
				}
				if (i != class_id && j == i) {
					trueNegative += (double) confusionMat.at<int>(i, j);
				}
			}
		}

		positive = truePositive+falseNegative;
		negative = falsePositive+trueNegative;

		specificity = trueNegative/negative;
		sensitivity = truePositive/positive;

		balancedAccuracyMean += (sensitivity + specificity)/(2.0);
	}

	return (balancedAccuracyMean/confusionMat.rows)*100.0;
}

/*******************************************************************************
Positive: minority class
Negative: majority class
*******************************************************************************/
vector<double> calculateFscore(Mat confusionMat){
	double fscore, truePositive, falseNegative, falsePositive;
	double trueNegative, precisionRate, recallRate, positive;
	vector<double> fScore;
	int class_id, i, j;

	for (class_id = 0; class_id < confusionMat.rows; class_id++){

		truePositive = falseNegative = falsePositive = trueNegative = 0;
		truePositive = confusionMat.at<int>(class_id, class_id);

		for(i = 0; i < confusionMat.rows; i++){
			for(j = 0; j < confusionMat.cols; j++){
				if(i == class_id && j != class_id) {
					falseNegative += confusionMat.at<int>(i, j);
				}
				if(i != class_id && j == class_id) {
					falsePositive += confusionMat.at<int>(i, j);
				}
				if(i != class_id && j == i) {
					trueNegative += confusionMat.at<int>(i, j);
				}
			}
		}

		positive = truePositive+falseNegative;

		precisionRate = truePositive/(truePositive+falsePositive);
		recallRate = truePositive/positive;

		fscore = (2.0) * (precisionRate*recallRate)/(precisionRate+recallRate);
		if (truePositive == 0) fScore.push_back(0.0);
		else fScore.push_back(fscore*100.0);
	}

	// falsePositiveRate = falsePositive/negative; // ROC: plotted on X axis
	// truePositiveRate = truePositive/positive; // ROC: plotted on Y axis == sensitivity

	return fScore;
}

/*******************************************************************************
Confusion Matrix
                          Predicted
actual class   truePositive   | falseNegative
               falsePositive  | trueNegative
*******************************************************************************/
Mat confusionMatrix(int numClasses, Mat labelsTesting, Mat result, int print) {
	int i, rightClass, guessedClass;
	Mat confusionMat = Mat::zeros(numClasses, numClasses, CV_32S);

	for (i = 0; i < result.size().height; i++) {
		rightClass = labelsTesting.at<int>(i,0);
		guessedClass = result.at<float>(i,0);
		confusionMat.at<int>(rightClass, guessedClass)++;
	}

	if (print) {
    cout << "---------------------------------------------------------" << endl;
    cout << "\t\t\t\tConfusion Matrix" << endl;
    cout << "\t\tPredicted" << endl << "\t";
    for(i = 0; i < confusionMat.cols; i++){
        cout << "\t" << i;
    }
    cout << endl;
    for(i = 0; i < confusionMat.rows; i++){
        if (i == 0) cout << "Real";
        cout << "\t"<< i;
        for(int j = 0; j < confusionMat.cols; j++){
            cout << "\t" << confusionMat.at<int>(i, j);
        }
        cout << endl;
    }
	}

	// /* Output file to use the confusion matrix plot with python*/
	// stringstream fileName, resultFile;
	// fileName << outputName+"_labels_" << minority.second << ".csv";
	// resultFile << outputName+"_resultlabels_" << minority.second << ".csv";
	// ofstream labels(fileName.str().c_str());
	// ofstream labelsResult(resultFile.str().c_str());
	// for (i = 0; i < result.size().height; i++) {
	//     labels << labelsTesting.at<int>(i, 0) << endl;
	//     labelsResult << result.at<float>(i, 0) << endl;
	// }
	// labels.close();
	// labelsResult.close();

	return confusionMat;
}

vector< vector<double> > Classifier::classify(double trainingRatio,
	int numRepetition, vector<Classes> imageClasses, string name, double id) {

	Mat result, confusionMat;
	int i, hits, num_features, rand_training, count_training, count_testing;
	int minorNumber, class_id, imgs_training, imgs_testing, pos, repetition, x;
	int minorityClass;
	std::vector<Classes>::iterator it;
	numClasses = imageClasses.size();
	vector<int> vectorRand, img_by_class(numClasses, 0), fixedSet(numClasses, 0);
	vector<int> trainingNumber(numClasses, 0), testingNumber(numClasses, 0);
	vector <vector<double> > fscore;
	vector<double> fscoreClasses;
	outputName = name;

	/* If training and testing set are fixed */
	for (it = imageClasses.begin(); it != imageClasses.end(); ++it) {
		img_by_class[it->classNumber] = it->features.size().height;
		num_features = it->features.size().width;
		for (i = 0; i < it->trainOrTest.size().height; i++) {
			fixedSet[it->classNumber] = 0;
			if (it->trainOrTest.at<int>(i,0) == 1) {
				trainingNumber[it->classNumber]++;
				fixedSet[it->classNumber] = 1;
			} else if (it->trainOrTest.at<int>(i,0) == 2) {
				testingNumber[it->classNumber]++;
				fixedSet[it->classNumber] = 1;
			}
		}
	}

	/* For each class we need to calculate the size of both training and testing sets, given a ratio */
	imgs_training = 0;
	imgs_testing = 0;
	for (i = 0; i < numClasses; i++) {
		if (trainingNumber[i] == 0) {
			trainingNumber[i] = ceil(img_by_class[i] * trainingRatio);
			testingNumber[i] = img_by_class[i] - trainingNumber[i];
		}
		imgs_testing += testingNumber[i];
		imgs_training += trainingNumber[i];
	}

	for (i = 0; i < numClasses; i++) {
    cout << "Number of images in class " << i << ": " << img_by_class[i] << endl;
    cout << "\tTraining: " << trainingNumber[i] << endl;
    cout << "\tTesting: " << testingNumber[i] << endl;
	}

	/* Repeated random sub-sampling validation */
	for (repetition = 0; repetition < numRepetition; repetition++) {
		Mat dataTraining(imgs_training, num_features, CV_32FC1);
		Mat labelsTraining(imgs_training, 1, CV_32S);
		Mat dataTesting(imgs_testing, num_features, CV_32FC1);
		Mat labelsTesting(imgs_testing, 1, CV_32S);
		count_training = 0;
		count_testing = 0;

		vectorRand.clear();

		for (it = imageClasses.begin(); it != imageClasses.end(); ++it) {
			if (fixedSet[it->classNumber]) {
				for (x = 0; x < it->trainOrTest.size().height; x++) {
					if (it->trainOrTest.at<int>(x,0) == 1) {
						Mat tmp = dataTraining.row(count_training);
						it->features.row(x).copyTo(tmp);
						labelsTraining.at<int>(count_training, 0) = it->classNumber;
						count_training++;
						vectorRand.push_back(x);
					}
				}
			}
			else {
				/* Generate a random position for each training data */
				rand_training = 0;
				while (rand_training < trainingNumber[it->classNumber]) {
					pos = rand() % img_by_class[it->classNumber];
					if (!count(vectorRand.begin(), vectorRand.end(), pos)) {
						Mat tmp = dataTraining.row(count_training);
						it->features.row(pos).copyTo(tmp);
						labelsTraining.at<int>(count_training, 0) = it->classNumber;
						rand_training++;
						count_training++;
						vectorRand.push_back(pos);
					}
				}
			}
			// After selecting the training set, the testing set it is going to be
			// the rest of the whole set
			for (i = 0; i < it->features.size().height; i++) {
				if (!count(vectorRand.begin(), vectorRand.end(), i)) {
					Mat tmp = dataTesting.row(count_testing);
					it->features.row(i).copyTo(tmp);
					labelsTesting.at<int>(count_testing, 0) = it->classNumber;
					count_testing++;
				}
			}
			vectorRand.clear();
		}


		/* Train and predict using the Normal Bayes classifier */
		//Classifier::bayesClassifier(dataTraining, labelsTraining, dataTesting, result);

		/* Train and predict using 1-KNN classifier */
		Classifier::knn(dataTraining, labelsTraining, dataTesting, 1, result);

		/* Counts how many samples were classified as expected */
		hits = 0;
		for (i = 0; i < result.size().height; i++) {
			if (labelsTesting.at<int>(i, 0) == result.at<float>(i, 0)) {
				hits++;
			}
		}
		totalTest = result.size().height;
		totalTrain = labelsTraining.size().height;
		accuracy.push_back((double)hits*100.0/(double)totalTest);

		confusionMat = confusionMatrix(numClasses, labelsTesting, result, 1);
		if (confusionMat.rows == 2){
			minorNumber = img_by_class[0];
			minorityClass = 0;
			for (class_id = 0; class_id < (int) img_by_class.size(); class_id++) {
				if (img_by_class[class_id] < minorNumber) {
					minorNumber = img_by_class[class_id];
					minorityClass = class_id;
				}
			}
			// This only pushes the minority class fscore
			vector<double> fscoreMinority;
			fscoreMinority.push_back(calculateFscore(confusionMat)[minorityClass]);
			fscore.push_back(fscoreMinority);
		}
		else {
			fscoreClasses = calculateFscore(confusionMat);
			if (fscore.size() == 0) {
				for (i = 0; i < numClasses; i++) {
					fscore.push_back(vector<double>()); // Add an empty row
				}
			}
			for (i = 0; i < (int) fscoreClasses.size(); i++) {
				if (fscoreClasses[i] > 0) {
					fscore[i].push_back(fscoreClasses[i]);
				}
			}
		}
		// precision.push_back(precisionRate/confusionMat.rows);
		// recall.push_back(recallRate/confusionMat.rows);

		balancedAccuracy.push_back(calculateBalancedAccuracy(confusionMat));

		dataTraining.release();
		dataTesting.release();
		labelsTesting.release();
		labelsTraining.release();
	}

	printAccuracy(id, fscore);
	accuracy.clear();
	balancedAccuracy.clear();
	precision.clear();
	recall.clear();

	return fscore;
}
