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
	cout <<  "Random cross validation with "<< accuracy.size() <<" k-fold:" << endl;
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

		outputFile.open((outputName+"BalancedAccuracy.csv").c_str(), ios::out | ios::app);
		outputFile << balancedMean << "\n";
		outputFile.close();

		outputFile.open((outputName+"FScore.csv").c_str(), ios::out | ios::app);
		outputFile << fscoreMean << "\n";
		// outputFile << fscoreMean << "\n";
		// for (i = 0; i < (int) fscores.size(); i++){
		// 	outputFile << fscores[i] << "\n";
		// }
		outputFile.close();
	}
}

/* Find which is the smaller class */
int Classifier::findSmallerClass(vector<ImageClass> imageClasses,
		int *minoritySize) {
	int class_id = 0, minorityClass;
	std::vector<ImageClass>::iterator it;
	// Number of images in the first class
	(*minoritySize) = imageClasses[0].images.size();

	// For each class
	for(it = imageClasses.begin(); it != imageClasses.end(); ++it) {
		// If the number of images in it is less than the minority number
		if (it->images.size() < (*minoritySize)) {
			(*minoritySize) = it->images.size();
			minorityClass = class_id;
		}
		class_id++;
	}
	return minorityClass;
}

double Classifier::calculateBalancedAccuracy(Mat confusionMat) {
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
vector<double> Classifier::calculateFscore(Mat confusionMat){
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
Mat Classifier::confusionMatrix(int numClasses, Mat labelsTesting, Mat result,
	int print) {

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
	int numRepetition, vector<ImageClass> data, string name, double id) {

	Mat result, confusionMat;
	int i, hits, numFeatures, numClasses, rep;
	vector <vector<double> > fscore;
	vector<double> fscoreClasses;
	outputName = name;
	std::vector<ImageClass>::iterator it;
	std::vector<Image>::iterator itImage;

	numClasses = data.numClasses();
	numFeatures = data.numFeatures();

	Mat dataTraining(0, numFeatures, CV_32FC1);
	Mat labelsTraining(0, 1, CV_32S);
	Mat dataTesting(0, numFeatures, CV_32FC1);
	Mat labelsTesting(0, 1, CV_32S);

	for (rep = 0; rep <= numRepetition; ++rep) {

		for (it = data.begin(); it != data.end(); ++it) {
			for (itImage = it->images.begin();
					itImage != it->images.end();
					++itImage) {
				if (data.isTraining(itImage->fold)) {
					dataTraining.push_back(itImage->features);
					labelsTraining.at<int>(dataTraining.rows-1, 0) = it->id;
				}
				else if (data.isTesting(itImage->fold)) {
					dataTesting.push_back(itImage->features);
					labelsTesting.at<int>(dataTesting.rows-1, 0) = it->id;
				}
			}
		}

		if (dataTraining.rows == 0) {
			for (it = data.begin(); it != data.end(); ++it) {

				numImages = it->images.size();
				numTraining = ceil(numImages * trainingRatio);

				/* Generate a random position for each training data */
				rand_training = 0;
				while (rand_training < numTraining) {
					pos = rand() % numImages;
					if (!count(vectorRand.begin(), vectorRand.end(), pos)) {
						dataTraining.push_back(it->images[pos].features);
						labelsTraining.at<int>(dataTraining.rows-1, 0) = it->id;
						rand_training++;
						vectorRand.push_back(pos);
					}
				}
				// After selecting the training set, the testing set it is going to be
				// the rest of the whole set
				for (i = 0; i < it->images.size(); i++) {
					if (!count(vectorRand.begin(), vectorRand.end(), i)) {
						dataTesting.push_back(it->images[pos].features);
						labelsTesting.at<int>(dataTesting.rows-1, 0) = it->id;
					}
				}
				vectorRand.clear();
			}
		}

		for (i = 0; i < numClasses; i++) {
	    cout << "\tTraining: " << dataTraining.rows << endl;
	    cout << "\tTesting: " << dataTesting.rows << endl;
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
		// }
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
