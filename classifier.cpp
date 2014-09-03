/**
 *
 *	Author: Gabriela Thumé
 *	Universidade de São Paulo / ICMC / 2014
 *
 **/

#include "classifier.h"

void Classifier::printAccuracy(vector<float> accuracy, int totalTest, int totalTrain, int smaller, int numClasses){

    float mean, variance, standardDeviation;
    int i;

    /* Calculate accuracy's mean */
    mean = 0;
    for (i = 0; i < (int) accuracy.size(); i++){
        mean += accuracy[i];
    }
    mean = mean/accuracy.size();

    /* Calculate accuracy's variance and std */
    variance = 0;
    for (i = 0; i < (int) accuracy.size(); i++){
        variance += pow(accuracy[i]-mean, 2);
    }
    variance = variance/accuracy.size();
    standardDeviation = sqrt(variance);

    cout << "\n---------------------------------------------------------------------------------------" << endl;
    cout << "Image classification using Normal Bayes classifier" << endl;
    cout << "--------------------------------------------------------------------------------------" << endl;
    cout << "Number of classes: " << numClasses << endl;
    cout << "Minority class samples: " << smaller << endl;
    cout << "Total samples: " << totalTest + totalTrain << " for each class: " << (totalTest + totalTrain)/numClasses;
    cout << endl << "Test samples: " << totalTest << " for each class: " << totalTest/numClasses << endl;
    cout << "Train samples: " << totalTrain << " for each class: " << totalTrain/numClasses << endl;
    cout <<  "Cross validation with "<<accuracy.size()<<" k-fold:" << endl;
    cout << "\tMean = " << mean << endl;
    cout << "\tVariance = " << variance << endl;
    cout << "\tStandard Deviation = " << standardDeviation << endl;
    cout << "---------------------------------------------------------------------------------------" << endl;

}

void Classifier::bayes(Mat vectorFeatures, Mat classes, int numClasses, float trainingRatio, int numRepetition){

    Mat result;
    CvNormalBayesClassifier classifier;
    int i, hits, height, width, trained, actual_class;
    int numTraining, num_testing, trainingSet, testingSet, totalTrain, totalTest, smaller;
    int totalTraining, start, pos, repetition;
    vector<float> accuracy;
    vector<int> vectorRand, dataClasse(numClasses, 0);
    srand(time(0));

    Size n = vectorFeatures.size();
    height = n.height;
    width = n.width;

    /* Count how many samples exists for each class */
    for(i = 0; i < height; i++){
        dataClasse[classes.at<float>(i,0)-1]++;
    }

    /* Find out what is the lowest number of samples */
    smaller = height+1;
    for(i = 0; i < numClasses; i++){
        if(dataClasse[i] < smaller){
            smaller = dataClasse[i];
        }
    }

    /* Calculate the size of both training and testing sets, given a ratio */
    trainingSet = (ceil(smaller*trainingRatio))*numClasses;
    testingSet = height-trainingSet;

    /* Repeated random sub-sampling validation */

    /* For each repetition, randomly divide the set into training and testing sets */
    totalTraining = trainingSet/numClasses;

    for(repetition = 0; repetition < numRepetition; repetition++) {

        Mat dataTraining(trainingSet, width, CV_32FC1);
        Mat labelsTraining(trainingSet, 1, CV_32FC1);
        Mat dataTesting(testingSet, width, CV_32FC1);
        Mat labelsTesting(testingSet, 1, CV_32FC1);

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

        totalTest = result.size().height;
        totalTrain = labelsTraining.size().height;
        accuracy.push_back(hits*100.0/totalTest);

        dataTraining.release();
        dataTesting.release();
        labelsTesting.release();
        labelsTraining.release();
    }

    printAccuracy(accuracy, totalTest, totalTrain, smaller, numClasses);

    accuracy.clear();
    classifier.clear();
}