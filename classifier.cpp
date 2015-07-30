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

    float mean, std, balancedMean, balancedStd, fscoreMean, fscoreStd;
    mean = accuracyMean(accuracy);
    std = standardDeviation(accuracy);
    balancedMean = accuracyMean(balancedAccuracy);
    balancedStd = standardDeviation(balancedAccuracy);
    fscoreMean = accuracyMean(fScore);
    fscoreStd = standardDeviation(fScore);
    // float R = accuracyMean(recall);
    // float P = accuracyMean(precision);
    ofstream outputFile;

    cout << "\n---------------------------------------------------------------------------------------" << endl;
    cout << "Image classification using KNN classifier" << endl;
    cout << "---------------------------------------------------------------------------------------" << endl;
    cout << "Number of classes: " << numClasses << endl;
    // cout << "Minority class samples: " << minority.second << endl;
    cout << "Total samples: " << totalTest + totalTrain << " for each class: +- " << (totalTest + totalTrain)/numClasses;
    cout << endl << "Test samples: " << totalTest << " for each class: +- " << totalTest/numClasses << endl;
    cout << "Train samples: " << totalTrain << " for each class: " << totalTrain/numClasses << endl;
    cout <<  "Cross validation with "<< accuracy.size() <<" k-fold:" << endl;
    cout << "\tMean Accuracy= " << mean << endl;
    cout << "\tStandard Deviation = " << std << endl;
    cout << "\tMean Balanced Accuracy= " << balancedMean << endl;
    cout << "\tStandard Deviation = " << balancedStd << endl;
    cout << "\tMean F1Score= " << fscoreMean << endl;
    cout << "\tStandard Deviation = " << fscoreStd << endl;
    cout << "---------------------------------------------------------------------------------------" << endl;

    if (outputName != ""){
        cout << "Write on " << (outputName+"FScore.csv").c_str() << endl;
    	cout << "---------------------------------------------------------------------------------------" << endl;
        // outputFile.open((outputName+"BalancedAccuracy.csv").c_str(), ios::out | ios::app);
        // outputFile << minority.second << "," << balancedMean << "\n";
        // outputFile.close();
        outputFile.open((outputName+"FScore.csv").c_str(), ios::out | ios::app);
        outputFile << minority.second << "," << fscoreMean << "\n";
        outputFile.close();
        // outputFile.open((outputName+"ROC.csv").c_str(), ios::out | ios::app);
        // outputFile << FPR << "," << TPR << endl;
        // outputFile.close();
    }
}

/* Find which is the smaller class and where it starts and ends */
void Classifier::findSmallerClass(Mat classes, int numClasses, int *smallerClass, int *start, int *end){

    int i, smaller;
    Size size = classes.size();
    vector<int> dataClasse(numClasses, 0);
    /* Discover the number of samples for each class */
    for(i = 0; i < size.height; i++){
        dataClasse[classes.at<float>(i,0)-1]++;
    }

    /* Find out which is the minority class */
    smaller = size.height +1;
    (*smallerClass) = -1;
    for(i = 0; i < (int) dataClasse.size(); i++){
        if(dataClasse[i] < smaller){
            smaller = dataClasse[i];
            (*smallerClass) = i;
        }
    }

    /* Where the minority class starts and ends */
    (*start) = -1;
    (*end) = -1;
    for(i = 0; i < size.height; i++){
        if(classes.at<float>(i,0)-1 == (*smallerClass)){
            if ((*start) == -1){
                (*start) = i;
            }
        }
        else if (*start != -1){
            (*end) = i;
            break;
        }
    }
    dataClasse.clear();
}

void Classifier::classify(float trainingRatio, int numRepetition, vector<Classes> imageClasses, string name = ""){

    Mat result, confusionMat;
    int i, hits, width, trained, numTraining, num_testing, classeId;
    int totalTraining = 0, totalTesting = 0, pos, repetition, actualClass = 0, x;
    int rightClass, guessedClass, higherClass, j, positiveClass;
    float truePositive , trueNegative, falsePositive, falseNegative, precisionRate;
    float specificity, sensitivity, balancedAccuracyMean, recallRate, score;
    float positive, negative, truePositiveRate, falsePositiveRate;
    srand(time(0));
    numClasses = imageClasses.size();
    vector<int> vectorRand, dataClasse(numClasses, 0), fixedSet(numClasses, 0), trainingNumber(numClasses, 0), testingNumber(numClasses, 0);
    outputName = name;

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

    for(i = 0; i < numClasses; i++){
        cout << "Number of images in class " << i << ": " << dataClasse[i] << endl;
        cout << "training images in class " << i << ": " << trainingNumber[i] << endl;
        cout << "testing images in class " << i << ": " << testingNumber[i] << endl;
    }

	/* For each class we need to calculate the size of both training and testing sets, given a ratio */
    for (i = 1; i <= numClasses; i++) {
    	actualClass = i-1;

        if (testingNumber[actualClass] == 0){
        	trainingNumber[actualClass] = (ceil(dataClasse[actualClass]*trainingRatio));
    		testingNumber[actualClass] = dataClasse[actualClass]-trainingNumber[actualClass];
        }
		totalTesting += testingNumber[actualClass];
		totalTraining += trainingNumber[actualClass];
		cout << "In class " << actualClass << " testing number: " << testingNumber[actualClass] << " training number: " << trainingNumber[actualClass] << endl;
	}

    cout << "Total of imagens in training " << totalTraining << " and in testing " << totalTesting << endl;

    /* Repeated random sub-sampling validation */
    for(repetition = 0; repetition < 1; repetition++) {
        Mat dataTraining(totalTraining, width, CV_32FC1);
        Mat labelsTraining(totalTraining, 1, CV_32FC1);
        Mat dataTesting(totalTesting, width, CV_32FC1);
        Mat labelsTesting(totalTesting, 1, CV_32FC1);
        numTraining = 0;

        vectorRand.clear();

        for(std::vector<Classes>::iterator it = imageClasses.begin(); it != imageClasses.end(); ++it) {

            cout << " classe " << it->classNumber << " fixed? " << fixedSet[it->classNumber] << endl;
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
                cout << " trainingNumber[it->classNumber] " << trainingNumber[it->classNumber] << endl;
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
            num_testing = 0;
            cout << " it->features.size().height " << it->features.size().height << endl;
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
        cout << "dataTraining " << dataTraining.size().height << " labelsTraining " << labelsTraining.size().height << " dataTesting " << dataTesting.size().height << " result " << result.size().height << endl;
        /* Counts how many samples were classified as expected */
        hits = 0;
        for (i = 0; i < result.size().height; i++) {
            if (labelsTesting.at<float>(i, 0) == result.at<float>(i, 0)){
                hits++;
            }
        }
        cout << "HITS " << hits << endl;
        totalTest = result.size().height;
        totalTrain = labelsTraining.size().height;
        accuracy.push_back(hits*100.0/totalTest);

        higherClass = labelsTraining.at<float>(labelsTraining.size().height-1,0);
        confusionMat = Mat::zeros(higherClass, numClasses, CV_32S);
        
        /* Confusion Matrix
								 Predicted
        actual class   truePositive   | falseNegative
           			   falsePositive  | trueNegative
        */
        
        for (i = 0; i < result.size().height; i++){
            rightClass = labelsTesting.at<float>(i,0);
            guessedClass = result.at<float>(i,0);
            confusionMat.at<int>(rightClass -1, guessedClass -1)++;
        }
        
        cout << "-------------------\nConfusion Matrix" << endl;
        for(i = 0; i < confusionMat.rows; i++){
            for(int j = 0; j < confusionMat.cols; j++){
                printf("%d\t", confusionMat.at<int>(i, j));
            }
            printf("\n");
        }
        
        // Calcular a media entre todas as classes dá uma ideia de todo o sistema
        // Dessa forma estamos olhando apenas para a classe minoritária em relação a todas as outras

        /* 
            Positive: minority class
            Negative: majority class
        */
        positiveClass = 0;
        if (confusionMat.rows == 2){
            positiveClass = actualClass;
        }

        score = balancedAccuracyMean = 0;
        for (classeId = positiveClass; classeId < confusionMat.rows; classeId++){

            truePositive = falseNegative = falsePositive = trueNegative = 0;
            truePositive = confusionMat.at<int>(classeId, classeId); /* minority correct */

            for(i = 0; i < confusionMat.rows; i++){
                for(j = 0; j < confusionMat.cols; j++){
                    if(i == classeId && j != classeId)
                        falseNegative += confusionMat.at<int>(i, j); /* minority predicted to be majority */
                    if(i != classeId && j == classeId)
                        falsePositive += confusionMat.at<int>(i, j); /* majority predicted to be minority */
                    if(i != classeId && j == i)
                        trueNegative += confusionMat.at<int>(i, j); /* majority correct */
                }
            }

            positive = truePositive+falseNegative;
            negative = falsePositive+trueNegative;

            specificity = trueNegative/negative;
            sensitivity = truePositive/positive;
            balancedAccuracyMean += (sensitivity + specificity)/2;
            // cout << "balanced mean " << balancedAccuracyMean << endl;
            precisionRate = truePositive/(truePositive+falsePositive);
            recallRate = truePositive/positive;
            score += 2.0 * (precisionRate*recallRate)/(precisionRate+recallRate);
            if (truePositive == 0)
                score = 0;
            // cout << "truePositive " << truePositive << endl;
            // cout << "falseNegative " << falseNegative << endl;
            // cout << "falsePositive " << falsePositive << endl;
            // cout << "score " << score << endl;

            // If there is only 2 classes, it is enough
            if (confusionMat.rows == 2){
                score *= 2;
                precisionRate *= 2;
                recallRate *= 2;
                balancedAccuracyMean *= 2;
                break;
            }
        }
        //cout << "score: " << score << endl;
        fScore.push_back((score/confusionMat.rows)*100.0);
        precision.push_back(precisionRate/confusionMat.rows);
        recall.push_back(recallRate/confusionMat.rows);
        balancedAccuracy.push_back((balancedAccuracyMean/confusionMat.rows)*100.0);

        falsePositiveRate = falsePositive/negative; // ROC: plotted on X axis
        truePositiveRate = truePositive/positive; // ROC: plotted on Y axis == sensitivity

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

        dataTraining.release();
        dataTesting.release();
        labelsTesting.release();
        labelsTraining.release();
    }
    
    printAccuracy();

    accuracy.clear();
    balancedAccuracy.clear();
    fScore.clear();
    precision.clear();
    recall.clear();
}