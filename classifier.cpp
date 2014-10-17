/**
 *
 *	Author: Gabriela Thumé
 *	Universidade de São Paulo / ICMC / 2014
 *
 **/

#include <iostream>
#include <fstream>
#include "classifier.h"

float accuracyMean(vector<float> accuracy){

    int i;
    float mean;
    /* Calculate accuracy's mean */
    mean = 0;
    for (i = 0; i < (int) accuracy.size(); i++){
        mean += accuracy[i];
    }
    mean = mean/accuracy.size();

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

    return std;
}

void Classifier::printAccuracy(){

    float mean, std, balancedMean, balancedStd;
    mean = accuracyMean(accuracy);
    std = standardDeviation(accuracy);
    balancedMean = accuracyMean(balancedAccuracy);
    balancedStd = standardDeviation(balancedAccuracy);

    ofstream outputFile;

    cout << "\n---------------------------------------------------------------------------------------" << endl;
    cout << "Image classification using Normal Bayes classifier" << endl;
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
        cout << "Write on " << (outputName+".csv").c_str();
        outputFile.open((outputName+".csv").c_str(), ios::out | ios::app);
        outputFile << minority.second;
        outputFile << ",";
        outputFile << balancedMean;
        outputFile << "\n";
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
    CvNormalBayesClassifier classifier;
    int i, hits, height, width, trained, actual_class;
    int numTraining, num_testing, trainingSet, testingSet;
    int totalTraining, start, pos, repetition;
    int rightClass, guessedClass, higherClass, j;
    float truePositive , trueNegative, falsePositive, falseNegative;
    float specificity, sensitivity, balancedAccuracyMean;
    srand(time(0));
    numClasses = nClasses;
    vector<int> vectorRand, dataClasse(numClasses, 0);
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

    /* Calculate the size of both training and testing sets, given a ratio */
    trainingSet = (ceil(minority.second*trainingRatio))*numClasses;
    testingSet = height-trainingSet;

    /* Repeated random sub-sampling validation */

    /* For each repetition, randomly divide the set into training and testing sets */
    totalTraining = trainingSet/numClasses;
    for(repetition = 0; repetition < numRepetition; repetition++) {

        Mat dataTraining(trainingSet, width, CV_32FC1);
        Mat labelsTraining(trainingSet, 1, CV_32FC1);
        Mat dataTesting(testingSet, width, CV_32FC1);
        Mat labelsTesting(testingSet, 1, CV_32FC1);
        truePositive = 0, trueNegative = 0, falsePositive = 0, falseNegative = 0;
        start = 0, numTraining = 0;

        for (i = 1; i <= numClasses; i++) {
            trained = 0;
            actual_class = i;

            /* Generate a random position for each training data */
            while (trained < totalTraining) {
                pos = start + (rand() % (dataClasse[actual_class-1]));
                if (!count(vectorRand.begin(), vectorRand.end(), pos)){
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
                Mat tmp = dataTesting.row(num_testing);
                vectorFeatures.row(i).copyTo(tmp);
                labelsTesting.at<float>(num_testing, 0) = classes.at<float>(i,0);
                num_testing++;
            }
        }
        vectorRand.clear();

        /* Train and predict using the Normal Bayes classifier */
        classifier.train(dataTraining, labelsTraining);
        classifier.predict(dataTesting, &result);

        /* Counts how many samples were classified as expected */
        hits = 0;
        for (i = 0; i < result.size().height; i++) {
            if (labelsTesting.at<float>(i, 0) == result.at<float>(i, 0)) {
                hits++;
            }
        }

        /*higherClass = labelsTraining.at<float>(labelsTraining.size().height-1,0);
        confusionMat = Mat::zeros(higherClass, numClasses, CV_32S);
        */
        /* Confusion Matrix
           truePositive   | falsePositive
           falseNegative  | trueNegative
        */
        /*
        for (i = 0; i< result.size().height; i++){
            rightClass = labelsTesting.at<float>(i,0)-1;
            guessedClass = result.at<float>(i,0)-1;
            confusionMat.at<int>(rightClass, guessedClass)++;
        }
        */
        /*cout << "-------------------\nConfusion Matrix" << endl;
        for(i = 0; i < confusionMat.rows; i++){
            for(int j = 0; j < confusionMat.cols; j++){
                printf("%d\t", confusionMat.at<int>(i, j));
            }
            printf("\n");
        }*/
        /*
        truePositive = confusionMat.at<int>(minority.first, minority.first);
        for(i = 0; i < confusionMat.rows; i++){
            for(j = 0; j < confusionMat.cols; j++){
                if(i == minority.first && j != minority.first)
                    falsePositive += confusionMat.at<int>(i, j);
                if(i != minority.first && j != i)
                    falseNegative += confusionMat.at<int>(i, j);
                if(i != minority.first && j == i)
                    trueNegative += confusionMat.at<int>(i, j);
            }
        }

        specificity = trueNegative/(trueNegative+falsePositive);
        sensitivity = truePositive/(truePositive+falseNegative);
        balancedAccuracyMean = (sensitivity + specificity)/2;
        balancedAccuracy.push_back(balancedAccuracyMean*100.0);
        */
        /* Output file to use the confusion matrix plot with python*/
        /*stringstream fileName;
        fileName << outputName+"_labels_";
        fileName << minority.second;
        fileName << ".csv";
        ofstream labels(fileName.str().c_str());
        for (i = 0; i < result.size().height; i++) {
            labels << labelsTesting.at<float>(i, 0) << ",";
            labels << result.at<float>(i, 0) << endl;
        }
        labels.close();*/

        totalTest = result.size().height;
        totalTrain = labelsTraining.size().height;
        accuracy.push_back(hits*100.0/totalTest);

        dataTraining.release();
        dataTesting.release();
        labelsTesting.release();
        labelsTraining.release();
    }

    /* When we run smote, the Mat features are balanced, so we previously know
    which is the minority class and its respective samples*/
    if(min.first != -1){
        minority = min;
    }

    printAccuracy();

    accuracy.clear();
    balancedAccuracy.clear();
    classifier.clear();
}
