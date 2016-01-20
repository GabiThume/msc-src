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
void Classifier::bayesClassifier(cv::Mat dataTraining, cv::Mat labelsTraining,
		cv::Mat dataTesting, cv::Mat& result) {
	CvNormalBayesClassifier classifier;
	classifier.train(dataTraining, labelsTraining);
	classifier.predict(dataTesting, &result);
	classifier.clear();
}

/* Train and predict using 1-KNN */
void Classifier::knn(cv::Mat dataTraining, cv::Mat labelsTraining,
    cv::Mat dataTesting, int k, cv::Mat& result) {
	cv::Mat responses, dist, nearests(1, k, CV_32F);
	CvKNearest knn(dataTraining, labelsTraining, responses, false, k);
	knn.find_nearest(dataTesting, k, result, nearests, dist);
	knn.clear();
	nearests.release();
	responses.release();
	dist.release();
}

double Classifier::calculateMean(std::vector<double> accuracy) {
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

double Classifier::calculateStandardDeviation(std::vector<double> accuracy) {
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

void Classifier::printAccuracy(double id,
    std::vector<std::vector<double> > fScore) {
	int i;
	std::ofstream outputFile;
	std::vector<double> fscores, fscoresStd;
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

  std::cout << std::endl;
	std::cout << "Total samples: " << totalTest + totalTrain << std::endl;
	std::cout << "\tTesting samples: " << totalTest << std::endl;
	std::cout << "\tTraining samples: " << totalTrain << std::endl;
	std::cout << "Random cross validation with " << accuracy.size();
  std::cout << " fold (k = 1 probably means that folds are fixed):";
  std::cout << std::endl;
	std::cout << "\tMean Accuracy: " << mean << std::endl;
	if (std != 0)
		std::cout << "\t\tStandard Deviation: " << std << std::endl;
	std::cout << "\tMean Balanced Accuracy: " << balancedMean << std::endl;
	if (balancedStd != 0)
		std::cout << "\t\tStandard Deviation: " << balancedStd << std::endl;
	std::cout << "\tMean F1Score: " << fscoreMean << std::endl;
	if (fscoreStd != 0)
		std::cout << "\t\tStandard Deviation: " << fscoreStd << std::endl;

	if (outputName != "") {

		// outputFile.open((outputName+"BalancedAccuracy.csv").c_str(),
    //                 std::ios::out | std::ios::app);
		// outputFile << balancedMean << "\n";
		// outputFile.close();

		outputFile.open((outputName+"FScore.csv").c_str(),
                    std::ios::out | std::ios::app);
		outputFile << fscoreMean << "\n";
		// outputFile << fscoreMean << "\n";
		// for (i = 0; i < (int) fscores.size(); i++){
		// 	outputFile << fscores[i] << "\n";
		// }
		outputFile.close();
    std::cout << "Wrote on " << (outputName+"FScore.csv").c_str() << std::endl;
	}
}

double Classifier::calculateBalancedAccuracy(cv::Mat confusionMat) {
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

	return (balancedAccuracyMean/(double)confusionMat.rows)*100.0;
}

/*******************************************************************************
Positive: minority class
Negative: majority class
*******************************************************************************/
std::vector<double> Classifier::calculateFscore(cv::Mat confusionMat) {
	double fscore, truePositive, falseNegative, falsePositive;
	double trueNegative, precisionRate, recallRate, positive;
	std::vector<double> fScore;
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
cv::Mat Classifier::confusionMatrix(int numClasses, cv::Mat labelsTesting,
    cv::Mat result, int print) {

	int i, rightClass, guessedClass;
	cv::Mat confusionMat = cv::Mat::zeros(numClasses, numClasses, CV_32S);

	for (i = 0; i < result.size().height; i++) {
		rightClass = labelsTesting.at<int>(i,0);
		guessedClass = result.at<float>(i,0);
		confusionMat.at<int>(rightClass, guessedClass)++;
	}

	if (print) {
    std::cout << std::endl;
    std::cout << "Confusion Matrix" << std::endl;
    std::cout << "\t\tPredicted" << std::endl << "\t";
    for(i = 0; i < confusionMat.cols; i++){
        std::cout << "\t" << i;
    }
    std::cout << std::endl;
    for(i = 0; i < confusionMat.rows; i++){
        if (i == 0) std::cout << "Real";
        std::cout << "\t"<< i;
        for(int j = 0; j < confusionMat.cols; j++){
            std::cout << "\t" << confusionMat.at<int>(i, j);
        }
        std::cout << std::endl;
    }
	}

	// /* Output file to use the confusion matrix plot with python*/
	// std::stringstream fileName, resultFile;
	// fileName << outputName+"_labels_" << minority.second << ".csv";
	// resultFile << outputName+"_resultlabels_" << minority.second << ".csv";
	// std::ofstream labels(fileName.str().c_str());
	// std::ofstream labelsResult(resultFile.str().c_str());
	// for (i = 0; i < result.size().height; i++) {
	//     labels << labelsTesting.at<int>(i, 0) << std::endl;
	//     labelsResult << result.at<float>(i, 0) << std::endl;
	// }
	// labels.close();
	// labelsResult.close();

	return confusionMat;
}

std::vector< std::vector<double> > Classifier::classify(double trainingRatio,
    int numRepetition, Data data, std::string name, double id) {

	cv::Mat result, confusionMat;
	int i, hits, numFeatures, numClasses, rep, numImages, numTraining;
	int pos, rand_training;
	std::vector<int> vectorRand;
	std::vector <std::vector<double> > fscore;
	std::vector<double> fscoreClasses;
	outputName = name;
	std::vector<ImageClass>::iterator it;
	std::vector<Image>::iterator itImage;

  std::cout << "-------------------------------------------------" << std::endl;
  std::cout << "Image classification using KNN classifier" << std::endl;

	numClasses = data.numClasses();
  std::cout << "Number of classes: " << numClasses << std::endl;
	numFeatures = data.numFeatures();
  std::cout << "Number of features: " << numFeatures << std::endl;

  cv::Mat dataTraining(0, numFeatures, CV_32FC1);
  cv::Mat labelsTraining(0, 1, CV_32S);
  cv::Mat dataTesting(0, numFeatures, CV_32FC1);
  cv::Mat labelsTesting(0, 1, CV_32S);

	for (rep = 0; rep < numRepetition; ++rep) {

    dataTraining.release();
  	labelsTraining.release();
  	dataTesting.release();
  	labelsTesting.release();

		for (it = data.classes.begin(); it != data.classes.end(); ++it) {
			for (itImage = it->images.begin();
					itImage != it->images.end();
					++itImage) {
				if (data.isTraining(it->id, itImage->fold)) {
					dataTraining.push_back(itImage->features);
					labelsTraining.push_back(it->id);
				}
				else if (data.isTesting(it->id, itImage->fold)) {
          dataTesting.push_back(itImage->features);
          labelsTesting.push_back(it->id);
				}
			}
		}

		if (dataTraining.rows == 0) {
			for (it = data.classes.begin(); it != data.classes.end(); ++it) {

				numImages = it->images.size();
				numTraining = ceil(numImages * trainingRatio);

				/* Generate a random position for each training data */
				rand_training = 0;
				while (rand_training < numTraining) {
					pos = rand() % numImages;
					if (!count(vectorRand.begin(), vectorRand.end(), pos)) {
						dataTraining.push_back(it->images[pos].features);
						labelsTraining.push_back(it->id);
						rand_training++;
						vectorRand.push_back(pos);
					}
				}
				// After selecting the training set, the testing set it is going to be
				// the rest of the whole set
				for (i = 0; i < (int) it->images.size(); i++) {
					if (!count(vectorRand.begin(), vectorRand.end(), i)) {
						dataTesting.push_back(it->images[pos].features);
						labelsTesting.push_back(it->id);
					}
				}
				vectorRand.clear();
			}
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
				fscore.push_back(std::vector<double>()); // Add an empty row
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

  std::cout << "Classification done." << std::endl;

	return fscore;
}
