/**
*
*	Author: Gabriela Thumé
*	Universidade de São Paulo / ICMC / 2014
*
**/

#include <iostream>
#include <fstream>
#include "classifier.h"

/* Train and predict using the Normal Bayes classifier */
void Classifier::bayesClassifier(Mat dataTraining, Mat labelsTraining, Mat dataTesting, Mat& result){
	CvNormalBayesClassifier classifier;
	classifier.train(dataTraining, labelsTraining);
	classifier.predict(dataTesting, &result);
	classifier.clear();
}

void Classifier::knn(Mat dataTraining, Mat labelsTraining, Mat dataTesting, Mat& result){
	int k = 1;
	Mat responses, dist, nearests(1, k, CV_32FC1);
	CvKNearest knn(dataTraining, labelsTraining, responses, false, k);
	knn.find_nearest(dataTesting, k, result, nearests, dist);
	knn.clear();
	nearests.release();
	responses.release();
	dist.release();
}

double Classifier::calculateMean(vector<double> accuracy){

	int i;
	double mean;
	/* Calculate accuracy's mean */
	mean = 0;
	for (i = 0; i < (int) accuracy.size(); i++){
		mean += accuracy[i];
	}
	mean = mean/accuracy.size();

	if (isnan(mean))
		mean = 0;

	return mean;
}

double Classifier::calculateStandardDeviation(vector<double> accuracy){

	int i;
	double mean, variance, std;
	mean = calculateMean(accuracy);
	/* Calculate accuracy's variance and std */
	variance = 0;
	for (i = 0; i < (int) accuracy.size(); i++){
		variance += pow(accuracy[i]-mean, 2);
	}
	variance = variance/accuracy.size();
	std = sqrt(variance);

	if (isnan(std))
		std = 0;

	return std;
}

void Classifier::printAccuracy(int id, vector<vector<double> > fScore){

	int i;
	ofstream outputFile;
	vector<double> fscores, fscoresStd;
	double mean, std, balancedMean, balancedStd, fscoreMean, fscoreStd;
	mean = calculateMean(accuracy);
	std = calculateStandardDeviation(accuracy);
	balancedMean = calculateMean(balancedAccuracy);
	balancedStd = calculateStandardDeviation(balancedAccuracy);

	for (i = 0; i < (int) fScore.size(); i++){
		cout << "Classe " << i << " FSCORE: " << calculateMean(fScore[i]) << endl;
		fscores.push_back(calculateMean(fScore[i]));
	}
	fscoreMean = calculateMean(fscores);

	for (i = 0; i < (int) fScore.size(); i++){
		fscoresStd.push_back(calculateStandardDeviation(fScore[i]));
	}
	fscoreStd = calculateStandardDeviation(fscoresStd);

	cout << "\n---------------------------------------------------------------------------------------" << endl;
	cout << "Image classification using KNN classifier" << endl;
	cout << "---------------------------------------------------------------------------------------" << endl;
	cout << "Number of classes: " << numClasses << endl;
	// cout << "Minority class samples: " << minority.second << endl;
	cout << "Total samples: " << totalTest + totalTrain;
	cout << endl << "\tTesting samples: " << totalTest << endl;
	cout << "\tTraining samples: " << totalTrain << endl;
	cout <<  "Cross validation with "<< accuracy.size() <<" k-fold:" << endl;
	cout << "\tMean Accuracy: " << mean << endl;
	if (std > 0)
	cout << "\t\tStandard Deviation: " << std << endl;
	cout << "\tMean Balanced Accuracy: " << balancedMean << endl;
	if (balancedStd > 0)
	cout << "\t\tStandard Deviation: " << balancedStd << endl;
	cout << "\tMean F1Score: " << fscoreMean << endl;
	if (fscoreStd > 0)
	cout << "\t\tStandard Deviation: " << fscoreStd << endl;
	cout << "---------------------------------------------------------------------------------------" << endl;

	if (outputName != ""){
		cout << "Write on " << (outputName+"FScore.csv").c_str() << endl;
		cout << "---------------------------------------------------------------------------------------" << endl;
		outputFile.open((outputName+"BalancedAccuracy.csv").c_str(), ios::out | ios::app);
		outputFile << balancedMean << "\n";
		// outputFile << minority.second << "," << balancedMean << "\n";
		outputFile.close();
		outputFile.open((outputName+"FScore.csv").c_str(), ios::out | ios::app);
		// outputFile.open((outputName+"FScore.csv").c_str());
		// outputFile << id << "," << fscoreMean << "\n";
		for (i = 0; i < (int) fscores.size(); i++){
			outputFile << i << "," << fscores[i] << "\n";
		}
		outputFile.close();
		// outputFile.open((outputName+"ROC.csv").c_str(), ios::out | ios::app);
		// outputFile << FPR << "," << TPR << endl;
		// outputFile.close();
	}
}

/* Find which is the smaller class */
int Classifier::findSmallerClass(vector<Classes> imageClasses){

	int classeId = 0, minorityClass, minorNumber;
	std::vector<Classes>::iterator it;

	for(it = imageClasses.begin(); it != imageClasses.end(); ++it) {
		if (it->features.size().height < minorNumber){
			minorNumber = it->features.size().height;
			minorityClass = classeId;
		}
		classeId++;
	}

	return minorityClass;
}

double calculateBalancedAccuracy(Mat confusionMat){

	double balancedAccuracyMean, truePositive, falseNegative, falsePositive;
	double trueNegative, specificity, sensitivity, positive, negative;
	int classeId, i, j;

	balancedAccuracyMean = 0;
	for (classeId = 0; classeId < confusionMat.rows; classeId++){

		truePositive = falseNegative = falsePositive = trueNegative = 0;
		truePositive = confusionMat.at<int>(classeId, classeId);

		for(i = 0; i < confusionMat.rows; i++){
			for(j = 0; j < confusionMat.cols; j++){
				if(i == classeId && j != classeId)
				falseNegative += confusionMat.at<int>(i, j);
				if(i != classeId && j == classeId)
				falsePositive += confusionMat.at<int>(i, j);
				if(i != classeId && j == i)
				trueNegative += confusionMat.at<int>(i, j);
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

/*
Positive: minority class
Negative: majority class
*/
vector<double> calculateFscore(Mat confusionMat){

	double fscore, truePositive, falseNegative, falsePositive;
	double trueNegative, precisionRate, recallRate, positive, negative;
	vector<double> fScore;
	int classeId, i, j;

	for (classeId = 0; classeId < confusionMat.rows; classeId++){

		truePositive = falseNegative = falsePositive = trueNegative = 0;
		truePositive = confusionMat.at<int>(classeId, classeId);

		for(i = 0; i < confusionMat.rows; i++){
			for(j = 0; j < confusionMat.cols; j++){
				if(i == classeId && j != classeId)
				falseNegative += confusionMat.at<int>(i, j);
				if(i != classeId && j == classeId)
				falsePositive += confusionMat.at<int>(i, j);
				if(i != classeId && j == i)
				trueNegative += confusionMat.at<int>(i, j);
			}
		}

		positive = truePositive+falseNegative;
		negative = falsePositive+trueNegative;

		precisionRate = truePositive/(truePositive+falsePositive);
		recallRate = truePositive/positive;

		fscore = (2.0) * (precisionRate*recallRate)/(precisionRate+recallRate);
		if (truePositive == 0)
		fScore.push_back(0);
		else
		fScore.push_back(fscore*100.0);
	}

	// falsePositiveRate = falsePositive/negative; // ROC: plotted on X axis
	// truePositiveRate = truePositive/positive; // ROC: plotted on Y axis == sensitivity

	return fScore;
}

/*
Confusion Matrix
Predicted
actual class   truePositive   | falseNegative
falsePositive  | trueNegative
*/
Mat confusionMatrix(int numClasses, Mat labelsTesting, Mat result){

	Mat confusionMat;
	int i, rightClass, guessedClass;

	confusionMat = Mat::zeros(numClasses, numClasses, CV_32S);

	for (i = 0; i < result.size().height; i++){
		rightClass = labelsTesting.at<float>(i,0);
		guessedClass = result.at<float>(i,0);
		confusionMat.at<int>(rightClass, guessedClass)++;
	}

	// if (print){
	//     cout << "---------------------------------------------------------------------------------------" << endl;
	//     cout << "\t\t\t\tConfusion Matrix" << endl;
	//     cout << "\t\tPredicted" << endl << "\t";
	//     for(i = 0; i < confusionMat.cols; i++){
	//         cout << "\t" << i;
	//     }
	//     cout << endl;
	//     for(i = 0; i < confusionMat.rows; i++){
	//         if (i == 0) cout << "Real";
	//         cout << "\t"<< i;
	//         for(int j = 0; j < confusionMat.cols; j++){
	//             cout << "\t" << confusionMat.at<int>(i, j);
	//         }
	//         cout << endl;
	//     }
	// }

	// /* Output file to use the confusion matrix plot with python*/
	// stringstream fileName, resultFile;
	// fileName << outputName+"_labels_" << minority.second << ".csv";
	// resultFile << outputName+"_resultlabels_" << minority.second << ".csv";
	// ofstream labels(fileName.str().c_str());
	// ofstream labelsResult(resultFile.str().c_str());
	// for (i = 0; i < result.size().height; i++) {
	//     labels << labelsTesting.at<float>(i, 0) << endl;
	//     labelsResult << result.at<float>(i, 0) << endl;
	// }
	// labels.close();
	// labelsResult.close();

	return confusionMat;
}

vector<vector<double> > Classifier::classify(double trainingRatio, int numRepetition, vector<Classes> imageClasses, string name, int id){

	Mat result, confusionMat;
	int i, hits, width, trained, numTraining, num_testing, minorNumber, classeId;
	int totalTraining = 0, totalTesting = 0, pos, repetition, actualClass = 0, x, minorityClass;
	numClasses = imageClasses.size();
	vector<int> vectorRand, dataClasse(numClasses, 0), fixedSet(numClasses, 0), trainingNumber(numClasses, 0), testingNumber(numClasses, 0);
	outputName = name;
	vector<vector<double> > fscore;
	vector<double> fscoreClasses;
	srand(time(0));

	/* If training and testing set are fixed */
	for(std::vector<Classes>::iterator it = imageClasses.begin(); it != imageClasses.end(); ++it) {
		dataClasse[it->classNumber] = it->features.size().height;
		width = it->features.size().width;
		for(i = 0; i < it->trainOrTest.size().height; i++){
			fixedSet[it->classNumber] = 0;
			if (it->trainOrTest.at<float>(i,0) == 1){
				trainingNumber[it->classNumber]++;
				fixedSet[it->classNumber] = 1;
			}
			if (it->trainOrTest.at<float>(i,0) == 2){
				testingNumber[it->classNumber]++;
				fixedSet[it->classNumber] = 1;
			}
		}
	}

	// for(i = 0; i < numClasses; i++){
	//     cout << "Number of images in class " << i << ": " << dataClasse[i] << endl;
	//     cout << "training images in class " << i << ": " << trainingNumber[i] << endl;
	//     cout << "testing images in class " << i << ": " << testingNumber[i] << endl;
	// }

	/* For each class we need to calculate the size of both training and testing sets, given a ratio */
	for (actualClass = 0; actualClass < numClasses; actualClass++) {
		if (trainingNumber[actualClass] == 0){
			trainingNumber[actualClass] = (ceil(dataClasse[actualClass]*trainingRatio));
			testingNumber[actualClass] = dataClasse[actualClass]-trainingNumber[actualClass];
		}
		totalTesting += testingNumber[actualClass];
		totalTraining += trainingNumber[actualClass];
		cout << "In class " << actualClass << " testing: " << testingNumber[actualClass] << " training: " << trainingNumber[actualClass] << endl;
	}

	cout << "Total of imagens in training " << totalTraining << " and in testing " << totalTesting << endl;

	/* Repeated random sub-sampling validation */
	for(repetition = 0; repetition < numRepetition; repetition++) {
		Mat dataTraining(totalTraining, width, CV_32FC1);
		Mat labelsTraining(totalTraining, 1, CV_32FC1);
		Mat dataTesting(totalTesting, width, CV_32FC1);
		Mat labelsTesting(totalTesting, 1, CV_32FC1);
		numTraining = 0;
		num_testing = 0;

		vectorRand.clear();

		for(std::vector<Classes>::iterator it = imageClasses.begin(); it != imageClasses.end(); ++it) {

			if (fixedSet[it->classNumber]) {
				for (x = 0; x < it->trainOrTest.size().height; x++){
					if (it->trainOrTest.at<float>(x,0) == 1){
						Mat tmp = dataTraining.row(numTraining);
						it->features.row(x).copyTo(tmp);
						labelsTraining.at<float>(numTraining, 0) = it->classNumber;
						trained++;
						numTraining++;
						vectorRand.push_back(x);
					}
				}
			}
			else {
				/* Generate a random position for each training data */
				trained = 0;
				while (trained < trainingNumber[it->classNumber]) {
					pos = rand() % dataClasse[it->classNumber];
					if (!count(vectorRand.begin(), vectorRand.end(), pos)){
						Mat tmp = dataTraining.row(numTraining);
						it->features.row(pos).copyTo(tmp);
						labelsTraining.at<float>(numTraining, 0) = it->classNumber;
						trained++;
						numTraining++;
						vectorRand.push_back(pos);
					}
				}
			}
			/* After selecting the training set, the testing set it is going to be the rest of the whole set */
			for (i = 0; i < it->features.size().height; i++) {
				if (!count(vectorRand.begin(), vectorRand.end(), i)){
					Mat tmp = dataTesting.row(num_testing);
					it->features.row(i).copyTo(tmp);
					labelsTesting.at<float>(num_testing, 0) = it->classNumber;
					num_testing++;
				}
			}
			vectorRand.clear();
		}


		/* Train and predict using the Normal Bayes classifier */
		//Classifier::bayesClassifier(dataTraining, labelsTraining, dataTesting, result);

		/* Train and predict using 1-KNN classifier */
		Classifier::knn(dataTraining, labelsTraining, dataTesting, result);
		/* Counts how many samples were classified as expected */
		hits = 0;
		for (i = 0; i < result.size().height; i++) {
			if (labelsTesting.at<float>(i, 0) == result.at<float>(i, 0)){
				hits++;
			}
		}
		totalTest = result.size().height;
		totalTrain = labelsTraining.size().height;
		accuracy.push_back(hits*100.0/totalTest);

		confusionMat = confusionMatrix(numClasses, labelsTesting, result);
		if (confusionMat.rows == 2){
			minorNumber = dataClasse[0];
			minorityClass = 0;
			for (classeId = 0; classeId < (int) dataClasse.size(); classeId++){
				if (dataClasse[classeId] < minorNumber){
					minorNumber = dataClasse[classeId];
					minorityClass = classeId;
				}
			}
			//fscore.push_back(calculateFscore(confusionMat, dataClasse)[minorityClass]);
		}
		else {
			fscoreClasses = calculateFscore(confusionMat);
			if (fscore.size() == 0) {
				for (int i = 0; i < numClasses; i++) {
					fscore.push_back(vector<double>()); // Add an empty row
				}
			}
			for (i = 0; i < (int) fscoreClasses.size(); i++) {
				if (fscoreClasses[i] > 0)
				fscore[i].push_back(fscoreClasses[i]);
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
