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
void bayesClassifier(Mat dataTraining, Mat labelsTraining, Mat dataTesting, Mat& result){
	CvNormalBayesClassifier classifier;
    classifier.train(dataTraining, labelsTraining);
    classifier.predict(dataTesting, &result);
    classifier.clear();
}

void knn(Mat dataTraining, Mat labelsTraining, Mat dataTesting, Mat& result){
	int k = 1;
	Mat responses, dist, nearests(1, k, CV_32FC1);
	CvKNearest knn(dataTraining, labelsTraining, responses, false, k);
	knn.find_nearest(dataTesting, k, result, nearests, dist);
	knn.clear();
}

float accuracyMean(vector<float> accuracy){

    int i;
    float mean;
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

float standardDeviation(vector<float> accuracy){

    int i;
    float mean, variance, std;
    mean = accuracyMean(accuracy);
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

void Classifier::printAccuracy(){

    float mean, std, balancedMean, balancedStd, FPR, TPR;
    mean = accuracyMean(accuracy);
    std = standardDeviation(accuracy);
    balancedMean = accuracyMean(balancedAccuracy);
    balancedStd = standardDeviation(balancedAccuracy);
    FPR = accuracyMean(falsePositiveRateValues);
    TPR = accuracyMean(truePositiveRateValues);

    ofstream outputFile;

    cout << "\n---------------------------------------------------------------------------------------" << endl;
    cout << "Image classification using KNN classifier" << endl;
    cout << "---------------------------------------------------------------------------------------" << endl;
    cout << "Number of classes: " << numClasses << endl;
    cout << "Minority class samples: " << minority.second << endl;
    cout << "Total samples: " << totalTest + totalTrain << " for each class: +- " << (totalTest + totalTrain)/numClasses;
    cout << endl << "Test samples: " << totalTest << " for each class: +- " << totalTest/numClasses << endl;
    cout << "Train samples: " << totalTrain << " for each class: " << totalTrain/numClasses << endl;
    cout <<  "Cross validation with "<< accuracy.size() <<" k-fold:" << endl;
    cout << "\tMean Accuracy= " << mean << endl;
    cout << "\tStandard Deviation = " << std << endl;
    cout << "\tMean Balanced Accuracy= " << balancedMean << endl;
    cout << "\tStandard Deviation = " << balancedStd << endl;
    cout << "---------------------------------------------------------------------------------------" << endl;

    if (outputName != ""){
        cout << "Write on " << (outputName+"BalancedAccuracy.csv").c_str() << endl;
    	cout << "---------------------------------------------------------------------------------------" << endl;
        outputFile.open((outputName+"BalancedAccuracy.csv").c_str(), ios::out | ios::app);
        outputFile << minority.second << "," << balancedMean << "\n";
        outputFile.close();
        outputFile.open((outputName+"TPR.csv").c_str(), ios::out | ios::app);
        outputFile << minority.second << "," << TPR*100.0 << "\n";
        outputFile.close();
        outputFile.open((outputName+"ROC.csv").c_str(), ios::out | ios::app);
        outputFile << FPR << "," << TPR << endl;
        outputFile.close();
    }
}

/* Find which is the smaller class and where it starts and ends */
void Classifier::findSmallerClass(Mat classes, int numClasses, int &smallerClass, int &start, int &end){

    int i, smaller;
    Size size = classes.size();
    vector<int> dataClasse(numClasses, 0);

    /* Discover the number of samples for each class */
    for(i = 0; i < size.height; i++){
        dataClasse[classes.at<float>(i,0)-1]++;
    }

    /* Find out which is the minority class */
    smaller = size.height +1;
    smallerClass = -1;
    for(i = 0; i < (int) dataClasse.size(); i++){
        if(dataClasse[i] < smaller){
            smaller = dataClasse[i];
            smallerClass = i;
        }
    }

    /* Where the minority class starts and ends */
    start = -1;
    end = -1;
    for(i = 0; i < size.height; i++){
        if(classes.at<float>(i,0)-1 == smallerClass){
            if (start == -1){
                start = i;
            }
        }
        else if (start != -1){
            end = i;
            break;
        }
    }
}

void Classifier::bayes(float trainingRatio, int numRepetition, Mat vectorFeatures, Mat classes, int nClasses, pair<int, int> min, string name = ""){

    Mat result, confusionMat;

    int i, hits, height, width, trained, actual_class, numTraining, num_testing;
    int totalTraining = 0, totalTesting = 0, start, pos, repetition, actualClass = 0;
    int rightClass, guessedClass, higherClass, j;
    float truePositive , trueNegative, falsePositive, falseNegative;
    float specificity, sensitivity, balancedAccuracyMean;
    float positive, negative, truePositiveRate, falsePositiveRate;
    srand(time(0));
    numClasses = nClasses;
    vector<int> vectorRand, dataClasse(numClasses, 0), trainingNumber(numClasses, 0), testingNumber(numClasses, 0);
    outputName = name;

    Size n = vectorFeatures.size();
    height = n.height;
    width = n.width;

    /* Count how many samples exists for each class */
    for(i = 0; i < height; i++){
        dataClasse[classes.at<float>(i,0)-1]++;
    }

    /* Find out what is the lowest number of samples */
    minority.first = 0;
    minority.second = height+1;
    for(i = 0; i < numClasses; i++){
        if(dataClasse[i] < minority.second){
            minority.first = i;
            minority.second = dataClasse[i];
        }
    }

    /* If it is rebalancing classification, min.first contains the minority class */
    if(min.first != -1){
    	minority.first = min.first;
        minority.second = min.second;
    }

	/* For each class we need to calculate the size of both training and testing sets, given a ratio */
    for (i = 1; i <= numClasses; i++) {
    	actualClass = i-1;
    	/* If the actual class is the one previously rebalanced, the training number is going to be the previously rebalanced minority */
    	if (minority.first == i)
			trainingNumber[actualClass] = minority.second;
		else
			trainingNumber[actualClass] = (ceil(dataClasse[actualClass]*trainingRatio));
		testingNumber[actualClass] = dataClasse[actualClass]-trainingNumber[actualClass];

		totalTesting += testingNumber[actualClass];
		totalTraining += trainingNumber[actualClass];
		//cout << " totalTesting " << totalTesting << " totalTraining " << totalTraining << " data class " << dataClasse[actualClass] << endl;
	}

    /* Repeated random sub-sampling validation */
    for(repetition = 0; repetition < 1; repetition++) {

        Mat dataTraining(totalTraining, width, CV_32FC1);
        Mat labelsTraining(totalTraining, 1, CV_32FC1);
        Mat dataTesting(totalTesting, width, CV_32FC1);
        Mat labelsTesting(totalTesting, 1, CV_32FC1);
        start = 0, numTraining = 0;

        vectorRand.clear();
        for (i = 1; i <= numClasses; i++) {
        	// cout << " CLASSE " << i << endl;
            trained = 0;
            actual_class = i-1;

            /* Generate a random position for each training data */
            while (trained < trainingNumber[actual_class]) {
            	/* If a minority class has been rebalanced, catch only the generated for training */
            	if (minority.first == i)
                	pos = start + (rand() % (trainingNumber[actual_class]));
                /* If not, randomly choose the training */
                else
                	pos = start + (rand() % (dataClasse[actual_class]));
                if (!count(vectorRand.begin(), vectorRand.end(), pos)){
                    // cout << ">>> treino " << pos << endl;
                    vectorRand.push_back(pos);
                    Mat tmp = dataTraining.row(numTraining);
                    vectorFeatures.row(pos).copyTo(tmp);
                    labelsTraining.at<float>(numTraining, 0) = classes.at<float>(pos,0);
                    trained++;
                    numTraining++;
                }
            }
            start += dataClasse[i-1];
        }

        /* After selecting the training set, the testing set it is going to be the rest of the whole set */
        num_testing = 0;
        for (i = 0; i < height; i++) {
            if (!count(vectorRand.begin(), vectorRand.end(), i)){
                // cout << ">>> teste " << i << endl;
                Mat tmp = dataTesting.row(num_testing);
                vectorFeatures.row(i).copyTo(tmp);
                labelsTesting.at<float>(num_testing, 0) = classes.at<float>(i,0);
                num_testing++;
            }
        }
        vectorRand.clear();

        /* Train and predict using the Normal Bayes classifier */
        //bayesClassifier(dataTraining, labelsTraining, dataTesting, result);
        knn(dataTraining, labelsTraining, dataTesting, result);

        /* Counts how many samples were classified as expected */
        hits = 0;
        for (i = 0; i < result.size().height; i++) {
            if (labelsTesting.at<float>(i, 0) == result.at<float>(i, 0)){
                hits++;
            }
        }

        totalTest = result.size().height;
        totalTrain = labelsTraining.size().height;
        cout << " hits " << hits << " total test " << totalTest << endl;
        accuracy.push_back(hits*100.0/totalTest);

        higherClass = labelsTraining.at<float>(labelsTraining.size().height-1,0);
        confusionMat = Mat::zeros(higherClass, numClasses, CV_32S);
        
        /* Confusion Matrix
								 Predicted
        actual class   truePositive   | falseNegative
           			   falsePositive  | trueNegative
        */
        
        for (i = 0; i< result.size().height; i++){
            rightClass = labelsTesting.at<float>(i,0)-1;
            guessedClass = result.at<float>(i,0)-1;
            confusionMat.at<int>(rightClass, guessedClass)++;
        }
        
        cout << "-------------------\nConfusion Matrix" << endl;
        for(i = 0; i < confusionMat.rows; i++){
            for(int j = 0; j < confusionMat.cols; j++){
                printf("%d\t", confusionMat.at<int>(i, j));
            }
            printf("\n");
        }

        /* This is only necessary because usually we have classes starting in 1, not in 0
        	it is going to be fixed later */
        actualClass = minority.first;
        if(min.first != -1)
        	actualClass--;
        
        /* 
			Positive: minority class
			Negative: majority class
        */
		truePositive = 0, falseNegative = 0, falsePositive =0, trueNegative = 0;
        truePositive = confusionMat.at<int>(actualClass, actualClass); /* minority correct */
        for(i = 0; i < confusionMat.rows; i++){
            for(j = 0; j < confusionMat.cols; j++){
                if(i == actualClass && j != actualClass)
                    falseNegative += confusionMat.at<int>(i, j); /* minority predicted to be majority */
                if(i != actualClass && j != i)
                    falsePositive += confusionMat.at<int>(i, j); /* majority predicted to be minority */
                if(i != actualClass && j == i)
                    trueNegative += confusionMat.at<int>(i, j); /* majority correct */
            }
        }

        positive = truePositive+falseNegative;
        negative = falsePositive+trueNegative;

        specificity = trueNegative/negative;
        sensitivity = truePositive/positive;
        balancedAccuracyMean = (sensitivity + specificity)/2;
        // if (!isnan(balancedAccuracyMean))
        // cout << " acuracia " << hits*100.0/totalTest << " accuracia balanceada " << balancedAccuracyMean*100.0 << " falsePositive " << falsePositive << endl;
        if (nClasses > 2)
        	balancedAccuracy.push_back(hits*100.0/totalTest);
        else
        	balancedAccuracy.push_back(balancedAccuracyMean*100.0);

        falsePositiveRate = falsePositive/negative; // ROC: plotted on X axis
        truePositiveRate = truePositive/positive; // ROC: plotted on Y axis == sensitivity

		truePositiveRateValues.push_back(truePositiveRate);
		falsePositiveRateValues.push_back(falsePositiveRate);

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


        /* TODO: Mahalanobis distance */

        dataTraining.release();
        dataTesting.release();
        labelsTesting.release();
        labelsTraining.release();
    }

    /* When we run smote, the Mat features are balanced, so we previously know
    which is the minority class and its respective samples*/
    if(min.first != -1){
        minority = min;
    	minority.second = testingNumber[minority.first-1];
    }
    else
    	minority.second = testingNumber[0];
    
    printAccuracy();

    accuracy.clear();
    balancedAccuracy.clear();
    truePositiveRateValues.clear();
    falsePositiveRateValues.clear();
}
